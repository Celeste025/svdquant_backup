#!/usr/bin/env python3
"""E027 fixed 4096-Q chunk baseline, existing native block-mean attention.

One real E026 input; no DiT, chunk search, graph capture, or output exactness gate.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E027'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E027/chunk')
N, NP, H, D, CHUNK = 22539, 22656, 56, 128, 4096


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    path = Path(path).absolute()
    return dict(file=str(path), bytes=path.stat().st_size, sha256=sha(path))


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def tensor_record(t):
    import torch
    t = t.detach().cpu().contiguous()
    return dict(shape=list(t.shape), dtype=str(t.dtype), finite=bool(t.float().isfinite().all()),
                sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy()).hexdigest())


def prepare_inputs(report):
    import torch
    reference = ROOT / 'results/research/E026/run.json'
    prior = json.loads(reference.read_text())
    assert prior['status'] == 'complete'
    refs = prior['inputs']
    artifacts = dict(capture=refs['capture'], q_packets=refs['q_packets'], kv_packets=refs['kv_packets'],
                     output=prior['outputs']['original']['artifact'])
    values = {}
    for name, ref in artifacts.items():
        assert record(ref['file']) == ref
        values[name] = torch.load(ref['file'], map_location='cpu', weights_only=True, mmap=True)
    raw = values['capture']
    assert all(raw[name].shape == (N, H, D) and raw[name].dtype == torch.bfloat16 for name in ('q', 'k', 'v'))
    assert raw['scale'] == D**-.5 and raw['valid_length'] == N
    assert tensor_record(values['output']) == prior['outputs']['original']['tensor']
    # The physical SF swizzle has independent 64-row slabs. Rebase offsets
    # exactly at each 4096-row boundary, including the 2176 padded-row tail.
    def offsets(rows):
        r, c = rows[:, None], torch.arange(D // 16)[None, :]
        return (r // 64) * 64 * (D // 16) + (c // 4) * 256 + (r % 16) * 16 + ((r % 64) // 16) * 4 + c % 4
    chunks = []
    for lo in range(0, NP, CHUNK):
        hi = min(lo + CHUNK, NP)
        assert lo % 128 == hi % 128 == 0
        assert torch.equal(offsets(torch.arange(lo, hi)) - lo * (D // 16), offsets(torch.arange(hi - lo)))
        chunks.append(dict(start=lo, padded_end=hi, valid_end=min(hi, N), query_groups=(hi - lo) // 128))
    report.update(reference_report=record(reference), inputs=artifacts, source_case=prior['source_case'],
        sources=dict(runner=record(__file__), official=record(prior['sources']['official']['file'])),
        geometry=dict(N=N, N_padded=NP, heads=H, head_dim=D, q_chunk=CHUNK, chunks=chunks),
        CPU_scale_slice_offsets_exact=True)
    assert report['sources']['official'] == prior['sources']['official']
    return values


def output_errors(value, reference):
    rows = []
    for head in range(H):
        a, b = value[:, head].double(), reference[:, head].double()
        assert bool(a.isfinite().all())
        error, energy = float((a - b).square().sum()), float(b.square().sum())
        rows.append(dict(head=head, error_energy=error, reference_energy=energy,
                         nmse=error / energy, max_abs=float((a - b).abs().max())))
    return dict(pooled_nmse=sum(r['error_energy'] for r in rows) / sum(r['reference_energy'] for r in rows),
                max_abs=max(r['max_abs'] for r in rows), per_head=rows,
                exactness_gate=False)


def run(args, report, values):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert args.deadline_unix and 0 < args.deadline_unix - time.time() <= 600
    assert torch.cuda.get_device_capability() == (12, 0)
    assert record(official.__file__) == report['sources']['official']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(min(1., 60 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    module = official.get_nvfp4_attention_sm120_module()
    counters = dict(preprocess=0, k_global_mean=0, q_pack=0, k_pack=0, v_pack=0, correction_gemm=0, attention=0)

    def budget():
        assert time.time() < args.deadline_unix, 'E027 supervisor deadline'
        assert torch.cuda.max_memory_allocated() <= 60 * 2**30, 'Observed allocator peak exceeds 60 GiB'

    def memory():
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())

    def preprocess_pack(q, k, v):
        # Precisely the official BF16 preprocessing order, stopping before its
        # full correction matmul. Native quantization kernels are reused below.
        counters['preprocess'] += 1
        counters['k_global_mean'] += 1
        k = k - k.mean(dim=-2, keepdim=True)
        q, k, v = map(official._pad_seq_len_to_128, (q, k, v))
        grouped = q.reshape(1, H, NP // 128, 128, D)
        mu = grouped.mean(dim=3)
        q = (grouped - mu.unsqueeze(3)).reshape(1, H, NP, D).contiguous()
        q4 = torch.empty((1, H, NP, D // 2), dtype=torch.uint8, device=q.device)
        k4 = torch.empty_like(q4)
        vt = torch.empty((1, H, D, NP // 2), dtype=torch.uint8, device=q.device)
        qs = torch.empty((1, H, NP, D // 16), dtype=torch.float8_e4m3fn, device=q.device)
        ks = torch.empty_like(qs)
        vs = torch.empty((1, H, D, NP // 16), dtype=torch.float8_e4m3fn, device=q.device)
        module.scaled_fp4_quant(q, q4, qs, 1); counters['q_pack'] += 1
        module.scaled_fp4_quant_permute(k, k4, ks, 1); counters['k_pack'] += 1
        module.scaled_fp4_quant_trans(v, vt, vs, 1); counters['v_pack'] += 1
        return (q4, k4, vt, qs, ks, vs), mu, k

    kwargs = dict(sm_scale=D**-.5, causal=False, per_block_mean=True, out_dtype=torch.bfloat16,
                  return_lse=False, unpadded_k_len=N)

    def consume(prepared, chunked):
        packets, mu, kc = prepared
        q4, k4, vt, qs, ks, vs = packets
        # One common K conversion per consumer, timed in both arms. Never
        # recenter/requantize K or V inside the chunk loop.
        kt = kc.transpose(-2, -1).float()
        if not chunked:
            correction = torch.matmul(mu.float(), kt).contiguous()
            counters['correction_gemm'] += 1
            result = official.nvfp4_attention_sm120_fwd(*packets, correction, **kwargs)
            counters['attention'] += 1
            return result[0, :, :N].transpose(0, 1).contiguous()
        assembled = torch.empty((N, H, D), device=q4.device, dtype=torch.bfloat16)
        for lo in range(0, NP, CHUNK):
            hi = min(lo + CHUNK, NP)
            # Physical Q/SF slabs are sliced at proven 64-row boundaries;
            # required contiguous copies are included in every measurement.
            qc, sf = q4[:, :, lo:hi].contiguous(), qs[:, :, lo:hi].contiguous()
            correction = torch.matmul(mu[:, :, lo // 128:hi // 128].float(), kt).contiguous()
            counters['correction_gemm'] += 1
            result = official.nvfp4_attention_sm120_fwd(qc, k4, vt, sf, ks, vs, correction, **kwargs)
            counters['attention'] += 1
            valid = min(hi, N) - lo
            assembled[lo:lo + valid].copy_(result[0, :, :valid].transpose(0, 1))
            del qc, sf, correction, result
        return assembled

    def upload():
        return tuple(values['capture'][key].transpose(0, 1).unsqueeze(0).contiguous().cuda() for key in ('q', 'k', 'v'))

    dense = upload()
    prepared = preprocess_pack(*dense)
    expected = {**values['q_packets'], **values['kv_packets']}
    report['packet_parity'] = {}
    for name, tensor in zip(('q_fp4', 'k_fp4', 'v_fp4_t', 'q_scale', 'k_scale', 'v_scale_t'), prepared[0]):
        got, want = tensor_record(tensor), tensor_record(expected[name])
        report['packet_parity'][name] = dict(actual=got, expected=want, exact=got == want)
        assert got == want, 'The baseline must preserve the source packet contract'
    report['mean_parity'] = tensor_record(prepared[1])
    reference = json.loads(Path(report['reference_report']['file']).read_text())
    assert report['mean_parity'] == reference['mean']
    del dense, tensor
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    DATA.mkdir(parents=True, exist_ok=False)
    report['arms'] = {}
    report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), gpu=0, tf32=False, cuda_graph=False)
    save(args.output, report)

    def benchmark(tier, mode, fn):
        budget(); gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        key = tier + '/' + mode
        entry = dict(status='warming', resident_baseline=memory(), warmup_calls=3, repeats=[],
                     counts_before=dict(counters))
        report['arms'][key] = entry
        save(args.output, report)
        for _ in range(3):
            budget(); output = fn(); del output
        torch.cuda.synchronize()
        entry.update(status='measuring', after_warmup_baseline=memory())
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for index in range(10):
            budget(); torch.cuda.reset_peak_memory_stats()
            before = memory()
            t0 = time.perf_counter(); begin.record()
            output = fn()
            end.record(); end.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000
            mem = memory()
            entry['repeats'].append(dict(index=index, cuda_ms=begin.elapsed_time(end), synchronized_wall_ms=wall_ms,
                before=before, after=mem, incremental_peak_allocated=mem['peak_allocated'] - before['allocated']))
            save(args.output, report)
            if index != 9:
                del output
        # Validation/CPU transfer, saving, hashes and all-head statistics are
        # strictly after measured end-event synchronization; no extra forward.
        cpu = output.cpu(); del output
        filename = DATA / (tier + '_' + mode + '.pt')
        torch.save(cpu, filename)
        entry.update(status='complete', output=dict(artifact=record(filename), tensor=tensor_record(cpu)),
            versus_E026_original_block=output_errors(cpu, values['output']),
            median_cuda_ms=statistics.median(r['cuda_ms'] for r in entry['repeats']),
            median_synchronized_wall_ms=statistics.median(r['synchronized_wall_ms'] for r in entry['repeats']),
            peak_allocated=max(r['after']['peak_allocated'] for r in entry['repeats']),
            peak_reserved=max(r['after']['peak_reserved'] for r in entry['repeats']),
            max_incremental_peak_allocated=max(r['incremental_peak_allocated'] for r in entry['repeats']),
            actual_counts={k: counters[k] - entry['counts_before'][k] for k in counters})
        assert entry['output']['tensor']['finite']
        expected_calls = 13 * (6 if mode == 'chunk4096' else 1)
        assert entry['actual_counts']['attention'] == entry['actual_counts']['correction_gemm'] == expected_calls
        for name in ('preprocess', 'k_global_mean', 'q_pack', 'k_pack', 'v_pack'):
            assert entry['actual_counts'][name] == (13 if tier == 'end_to_end' else 0)
        save(args.output, report)
        print(json.dumps(dict(arm=key, cuda_ms=entry['median_cuda_ms'], wall_ms=entry['median_synchronized_wall_ms'],
                              peak_allocated_gib=entry['peak_allocated']/2**30)), flush=True)

    benchmark('packed_consumer', 'full', lambda: consume(prepared, False))
    benchmark('packed_consumer', 'chunk4096', lambda: consume(prepared, True))
    del prepared
    gc.collect(); torch.cuda.empty_cache()
    dense = upload()
    benchmark('end_to_end', 'full', lambda: consume(preprocess_pack(*dense), False))
    benchmark('end_to_end', 'chunk4096', lambda: consume(preprocess_pack(*dense), True))
    report.update(status='complete', actual_total_counts=counters, complete_dit_calls=0,
        correction_storage=dict(full_bytes=H*(NP//128)*NP*4, max_chunk_bytes=H*(CHUNK//128)*NP*4),
        cuda_initialized=True)
    budget()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'run'), required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or RD / ('chunk_' + args.phase + '.json')
    assert not args.output.exists(), 'Preserve prior attempt'
    report = dict(experiment='E027', phase=args.phase, status='running', complete_dit_calls=0,
        policy='Full vs fixed Q4096; three warmups and ten repeats per arm at each of two scopes. Same native kernel, no output byte-equality acceptance gate.',
        timing_scope='CUDA events plus synchronized host wall; correction GEMM, required packet copies, attention launches, and valid NHD output assembly included. End-to-end adds exactly one common QKV preprocess/pack per call. Inputs already resident; validation/CPU copies/files excluded.',
        memory_scope='Each arm records resident and warmed baselines, total peaks, and incremental allocated peaks. Peak reserved reflects allocator history after arm-local cache clearing; no CUDA graph.',
        preprocessing='Official order and quantizer kernels, but no call to _preprocess_qkv/quantize_qkv and no full correction in the chunk preparation path.')
    started = time.time()
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase == 'check':
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        values = prepare_inputs(report)
        if args.phase == 'check':
            assert not torch.cuda.is_initialized()
            report.update(status='complete', cuda_initialized=False)
        else:
            prior = json.loads((RD / 'chunk_check.json').read_text())
            assert prior['status'] == 'complete' and prior['sources'] == report['sources'] and prior['inputs'] == report['inputs']
            with torch.inference_mode():
                run(args, report, values)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.time() - started
        save(args.output, report)
        print(json.dumps(dict(status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
