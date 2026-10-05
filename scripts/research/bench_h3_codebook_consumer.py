#!/usr/bin/env python3
"""E033: ten fixed-input consumer controls, then 468 full-path native calls.

Only the three saved H3 N22539/D128 cases. No DiT, graph, parameter sweep, or
general backend. Codebook construction is the unchanged E032 K16/six-Lloyd
baseline; private table lookup is compared with the old gathered consumer.
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

import probe_h3_query_mean_k4 as common
import probe_h3_codebook_centers as codebook
import probe_h3_adaptive_centers as adaptive

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E033'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E033')
H, N, NP, D, G = 56, 22539, 22656, 128, 177
ARMS = ('global', 'fullblock', 'chunk4096', 'chunk8192', 'codebook')
CALLS = dict(global_=1, fullblock=1, chunk4096=6, chunk8192=3, codebook=1)
CALLS['global'] = CALLS.pop('global_')
PACKET_NAMES = ('q_fp4', 'k_fp4', 'v_fp4_t', 'q_scale', 'k_scale', 'v_scale_t')


def trecord(t):
    import torch
    t = t.detach().cpu().contiguous()
    return dict(shape=list(t.shape), dtype=str(t.dtype), finite=bool(t.float().isfinite().all()),
                sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def chunk_geometry(size):
    """E027's physical SF proof, also applied to the fixed 8192-row partition."""
    import torch
    def offsets(rows):
        r, c = rows[:, None], torch.arange(D // 16)[None, :]
        return ((r // 64) * 64 * (D // 16) + (c // 4) * 256
                + (r % 16) * 16 + ((r % 64) // 16) * 4 + c % 4)
    rows = []
    for lo in range(0, NP, size):
        hi = min(lo + size, NP)
        assert lo % 128 == hi % 128 == 0
        assert torch.equal(offsets(torch.arange(lo, hi)) - lo * (D // 16),
                           offsets(torch.arange(hi - lo)))
        rows.append(dict(start=lo, padded_end=hi, valid_end=min(hi, N)))
    return rows


def binding(report, budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    import codebook_attention_sm120 as private
    path = ROOT / 'results/research/E032/run.json'
    previous = json.loads(path.read_text())
    assert previous['status'] == 'complete' and previous['attention_calls'] == 3
    assert previous['clusters'] == codebook.K == 16
    assert previous['lloyd_iterations'] == codebook.ITERATIONS == 6
    for key, source in (('runner', codebook.__file__), ('common', common.__file__),
                        ('previous', adaptive.__file__), ('official', official.__file__)):
        assert common.record(source) == previous['sources'][key], key
    report['sources'] = {name: common.record(p) for name, p in dict(
        runner=__file__, codebook=codebook.__file__, common=common.__file__,
        adaptive=adaptive.__file__, official=official.__file__, private=private.__file__,
        chunk_reference=ROOT/'scripts/research/bench_h3_chunked_correction.py').items()}
    report['reference_report'] = common.record(path)
    report['private_source'] = private.prepare_private_source()  # CPU only; never JIT here.
    report['inputs'] = []
    for block in (0, 24, 48):
        budget()
        source = next(s for s in previous['inputs'] if s['block'] == block)
        layer = next(s for s in previous['layers'] if s['block'] == block)
        raw, kv, centers = map(common.load, (source['capture'], source['kv_packets'], layer['centers']))
        for name in ('q', 'k', 'v'):
            assert raw[name].shape == (N, H, D) and raw[name].dtype == torch.bfloat16
        assert raw['valid_length'] == N and raw['scale'] == D**-.5
        assert centers['centers'].shape == (H, 16, D) and centers['centers'].dtype == torch.bfloat16
        assert centers['ids'].shape == (H, G) and centers['ids'].dtype == torch.int32
        assert int(centers['ids'].min()) >= 0 and int(centers['ids'].max()) < 16
        assert centers['mu'].shape == (H, G, D) and centers['global_mu'].shape == (H, 1, D)
        assert kv['k_fp4'].shape == (1, H, NP, D//2)
        assert kv['k_scale'].shape == (1, H, NP, D//16)
        assert kv['v_fp4_t'].shape == (1, H, D, NP//2)
        assert kv['v_scale_t'].shape == (1, H, D, NP//16)
        refs = dict(bf16=source['outputs']['bf16'], original_block=source['outputs']['original_block'],
                    original_global=source['outputs']['original_global'], codebook=layer['output'])
        for ref in refs.values():
            value = common.load(ref['artifact'])
            assert value.shape == (N, H, D) and value.dtype == torch.bfloat16
        report['inputs'].append(dict(block=block, capture=source['capture'], kv_packets=source['kv_packets'],
            centers=layer['centers'], center_tensors={k: trecord(v) for k, v in centers.items()}, outputs=refs))
        del raw, kv, centers, value
    report['chunks'] = {str(size): chunk_geometry(size) for size in (4096, 8192)}
    report['CPU_scale_slice_offsets_exact'] = True


def numeric(value, reference, ulp=False):
    """CPU FP64 head metrics; optional BF16 code distance with +/-0 merged."""
    import torch
    assert value.shape == reference.shape == (N, H, D)
    assert value.dtype == reference.dtype == torch.bfloat16
    rows = []
    for head in range(H):
        va, vb = value[:, head], reference[:, head]
        a, b = va.double(), vb.double()
        assert bool(a.isfinite().all()) and bool(b.isfinite().all())
        difference = a-b
        row = dict(head=head, error_energy=float(difference.square().sum()),
                   reference_energy=float(b.square().sum()), max_abs=float(difference.abs().max()),
                   elements=a.numel())
        row['nmse'] = row['error_energy']/row['reference_energy'] if row['reference_energy'] else None
        if ulp:
            # BF16 sign-magnitude codes: monotone -magnitude / +magnitude.
            # Both zero encodings map to 0; finite values were checked above.
            def ordered(t):
                bits = t.contiguous().view(torch.int16).to(torch.int32)
                magnitude = bits & 0x7fff
                return torch.where(bits < 0, -magnitude, magnitude)
            distance = (ordered(va)-ordered(vb)).abs()
            row.update(max_ulp=int(distance.max()), count_gt1_ulp=int((distance > 1).sum()),
                       count_nonzero_ulp=int((distance != 0).sum()))
        rows.append(row)
    error = sum(r['error_energy'] for r in rows)
    energy = sum(r['reference_energy'] for r in rows)
    result = dict(pooled_nmse=error/energy if energy else None, max_abs=max(r['max_abs'] for r in rows),
                  error_energy=error, reference_energy=energy, per_head=rows)
    if ulp:
        result.update(max_ulp=max(r['max_ulp'] for r in rows),
                      count_gt1_ulp=sum(r['count_gt1_ulp'] for r in rows),
                      count_nonzero_ulp=sum(r['count_nonzero_ulp'] for r in rows),
                      signed_zeros_merged=True, engineering_limit_ulp=1)
    return result


def run(args, report, budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    import codebook_attention_sm120 as private
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(min(1., 60*2**30/torch.cuda.get_device_properties(0).total_memory))
    for name in ('validation', 'benchmark'):
        assert not (DATA/name).exists(), 'Preserve prior outputs'
        (DATA/name).mkdir(parents=True)
    report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), visible_gpu=0, tf32=False, cuda_graph=False)
    common.save(args.output, report)
    budget()
    official_module = official.get_nvfp4_attention_sm120_module()
    private.get_codebook_module()  # Compilation is inside the shared deadline, outside timers/call count.
    budget()
    weights = torch.full((1, G), 128., device='cuda', dtype=torch.float32)
    weights[:, -1] = 11
    names = ('preprocess', 'k_mean', 'q_block_mean', 'q_global_mean', 'construct',
             'q_pack', 'k_pack', 'v_pack', 'correction_gemm', 'full_correction_gather',
             'official_attention', 'private_attention', 'attention')
    counters = {name: 0 for name in names}
    report['actual_counts'] = counters
    context = {'stage': 'validation'}

    def memory():
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())

    def check_memory():
        budget()
        assert torch.cuda.max_memory_allocated() <= 60*2**30, 'Observed allocation exceeds 60 GiB'

    def native(packets, table, ids=None, global_mode=False):
        field = 'validation_attention_calls' if context['stage'] == 'validation' else 'benchmark_attention_calls'
        limit = 10 if context['stage'] == 'validation' else 468
        assert report[field] < limit
        report[field] += 1
        counters['attention'] += 1
        if ids is not None:
            counters['private_attention'] += 1
            return private.codebook_fwd(*packets, table, ids, sm_scale=D**-.5, unpadded_k_len=N)
        counters['official_attention'] += 1
        return official.nvfp4_attention_sm120_fwd(*packets, table, sm_scale=D**-.5,
            causal=False, per_block_mean=not global_mode, out_dtype=torch.bfloat16,
            return_lse=False, unpadded_k_len=N)

    def pack_q(qcenter):
        q4 = torch.empty((1, H, NP, D//2), device='cuda', dtype=torch.uint8)
        qs = torch.empty((1, H, NP, D//16), device='cuda', dtype=torch.float8_e4m3fn)
        official_module.scaled_fp4_quant(qcenter.unsqueeze(0), q4, qs, 1)
        counters['q_pack'] += 1
        return q4, qs

    def output_record(value, path):
        torch.save(value, path)
        return dict(artifact=common.record(path), tensor=trecord(value))

    def validate_pair(block, mode, packets, table, ids, endpoint=None):
        check_memory()
        private.validate_center_ids(ids, table.shape[-2])  # Value scan only outside timers.
        entry = dict(block=block, mode=mode, status='running', counts_before=dict(counters),
            inputs={name: trecord(t) for name, t in zip(PACKET_NAMES, packets)},
            table=trecord(table), ids=trecord(ids), outputs={})
        report['validation'].append(entry)
        common.save(args.output, report)
        # Same packed tensors and table are used by both calls.
        new = native(packets, table, ids=ids)[0, :, :N].transpose(0, 1).contiguous().cpu()
        entry['outputs']['private'] = output_record(new, DATA/'validation'/f'block{block}_{mode}_private.pt')
        if mode == 'zero':
            expanded = table  # Old global branch broadcasts this one correction row.
        elif mode == 'identity':
            expanded = table  # arange ids; already one row for every original Q group.
        else:
            expanded = table.gather(2, ids.long().unsqueeze(-1).expand(-1, -1, -1, NP)).contiguous()
            counters['full_correction_gather'] += 1
        old = native(packets, expanded, global_mode=mode == 'zero')[0, :, :N].transpose(0, 1).contiguous().cpu()
        entry['outputs']['old'] = output_record(old, DATA/'validation'/f'block{block}_{mode}_old.pt')
        entry['private_vs_same_input_old'] = numeric(new, old, ulp=True)
        if endpoint is not None:
            reference = common.load(endpoint['artifact'])
            entry['old_vs_saved_recipe'] = numeric(old, reference)
        entry['actual_counts'] = {k: counters[k]-entry['counts_before'][k] for k in counters}
        entry['status'] = 'complete'
        common.save(args.output, report)
        del new, old, expanded

    # Fixed C/id validation uses saved E032 centers, with actual Q/T rebuilt.
    for source in report['inputs']:
        check_memory()
        raw = common.load(source['capture'])
        saved = common.load(source['centers'])
        kv = common.load(source['kv_packets'])
        q, k = (raw[key].transpose(0, 1).contiguous().cuda() for key in ('q', 'k'))
        qpad, kc, mu, global_mu = adaptive.means_and_key(q, k)
        centers, ids = saved['centers'].cuda(), saved['ids'].unsqueeze(0).contiguous().cuda()
        selected = centers.gather(1, ids[0].long().unsqueeze(-1).expand(-1, -1, D))
        qcenter = (qpad.reshape(H, G, 128, D)-selected.unsqueeze(2)).reshape(H, NP, D).contiguous()
        table = (centers.float()@kc.transpose(-2, -1).float()).unsqueeze(0).contiguous()
        counters['correction_gemm'] += 1
        q4, qs = pack_q(qcenter)
        gpu_kv = {name: value.cuda() for name, value in kv.items()}
        packets = (q4, gpu_kv['k_fp4'], gpu_kv['v_fp4_t'], qs, gpu_kv['k_scale'], gpu_kv['v_scale_t'])
        validate_pair(source['block'], 'codebook', packets, table, ids, source['outputs']['codebook'])
        del packets, table, qcenter, q4, qs, centers, ids, selected
        if source['block'] == 0:
            for mode, center in (('identity', mu), ('zero', global_mu)):
                if mode == 'identity':
                    ids = torch.arange(G, device='cuda', dtype=torch.int32)[None, None].expand(1, H, G).contiguous()
                    qcenter = (qpad.reshape(H, G, 128, D)-center.unsqueeze(2)).reshape(H, NP, D).contiguous()
                else:
                    ids = torch.zeros((1, H, G), device='cuda', dtype=torch.int32)
                    qcenter = (qpad-center).contiguous()
                table = (center.float()@kc.transpose(-2, -1).float()).unsqueeze(0).contiguous()
                counters['correction_gemm'] += 1
                q4, qs = pack_q(qcenter)
                packets = (q4, gpu_kv['k_fp4'], gpu_kv['v_fp4_t'], qs, gpu_kv['k_scale'], gpu_kv['v_scale_t'])
                endpoint = source['outputs']['original_block' if mode == 'identity' else 'original_global']
                validate_pair(0, mode, packets, table, ids, endpoint)
                del ids, table, qcenter, q4, qs, packets
            del center
        del raw, saved, kv, q, k, qpad, kc, mu, global_mu, gpu_kv
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    assert report['validation_attention_calls'] == 10
    report['validation_counts'] = dict(counters)
    bad = [dict(block=r['block'], mode=r['mode'], max_ulp=r['private_vs_same_input_old']['max_ulp'],
                count_gt1_ulp=r['private_vs_same_input_old']['count_gt1_ulp'])
           for r in report['validation'] if r['private_vs_same_input_old']['count_gt1_ulp']]
    report['consumer_control'] = dict(passed=not bad, violations=bad, finite_bf16_max_ulp=1,
        meaning='Engineering addressing/compilation control on identical packets/table; not model-quality gate or scientific negative.')
    if bad:
        report.update(status='stop_before_benchmark', cuda_initialized=True)
        common.save(args.output, report)
        return
    context['stage'] = 'benchmark'

    def prepare_pack(dense, mode):
        q, k, v = dense  # resident contiguous HND BF16 inputs
        counters['preprocess'] += 1
        counters['k_mean'] += 1
        ids = None
        if mode == 'codebook':
            qpad, kc, mu, global_mu = adaptive.means_and_key(q, k)
            counters['q_block_mean'] += 1; counters['q_global_mean'] += 1
            result = codebook.construct(mu, global_mu, weights)
            counters['construct'] += 1
            centers = result['centers']
            ids = result['ids'].unsqueeze(0).contiguous()
            selected = centers.gather(1, result['ids'].long().unsqueeze(-1).expand(-1, -1, D))
            qcenter = (qpad.reshape(H, G, 128, D)-selected.unsqueeze(2)).reshape(H, NP, D).contiguous()
        else:
            kc = torch.nn.functional.pad(k-k.mean(-2, keepdim=True), (0, 0, 0, NP-N))
            qpad = torch.nn.functional.pad(q, (0, 0, 0, NP-N))
            if mode == 'global':
                centers = qpad.mean(-2, keepdim=True); counters['q_global_mean'] += 1
                qcenter = (qpad-centers).contiguous()
            else:
                centers = qpad.reshape(H, G, 128, D).mean(2); counters['q_block_mean'] += 1
                qcenter = (qpad.reshape(H, G, 128, D)-centers.unsqueeze(2)).reshape(H, NP, D).contiguous()
        vpad = torch.nn.functional.pad(v, (0, 0, 0, NP-N))
        q4, qs = pack_q(qcenter)
        k4, ks = torch.empty_like(q4), torch.empty_like(qs)
        vt = torch.empty((1, H, D, NP//2), device='cuda', dtype=torch.uint8)
        vs = torch.empty((1, H, D, NP//16), device='cuda', dtype=torch.float8_e4m3fn)
        official_module.scaled_fp4_quant_permute(kc.unsqueeze(0), k4, ks, 1); counters['k_pack'] += 1
        official_module.scaled_fp4_quant_trans(vpad.unsqueeze(0), vt, vs, 1); counters['v_pack'] += 1
        return (q4, k4, vt, qs, ks, vs), centers, kc, ids

    def execute(dense, mode):
        packets, centers, kc, ids = prepare_pack(dense, mode)
        # Exactly one K-to-FP32 conversion per entire invocation, outside chunk loop.
        kt = kc.transpose(-2, -1).float()
        if not mode.startswith('chunk'):
            table = (centers.float()@kt).unsqueeze(0).contiguous()
            counters['correction_gemm'] += 1
            result = native(packets, table, ids=ids, global_mode=mode == 'global')
            return result[0, :, :N].transpose(0, 1).contiguous()
        size = int(mode.removeprefix('chunk'))
        q4, k4, vt, qs, ks, vs = packets
        assembled = torch.empty((N, H, D), device='cuda', dtype=torch.bfloat16)
        for piece in report['chunks'][str(size)]:
            lo, hi = piece['start'], piece['padded_end']
            qc, sf = q4[:, :, lo:hi].contiguous(), qs[:, :, lo:hi].contiguous()
            correction = (centers[:, lo//128:hi//128].float()@kt).unsqueeze(0).contiguous()
            counters['correction_gemm'] += 1
            result = native((qc, k4, vt, sf, ks, vs), correction)
            valid = piece['valid_end']-lo
            assembled[lo:lo+valid].copy_(result[0, :, :valid].transpose(0, 1))
            del qc, sf, correction, result
        return assembled

    def benchmark(source, dense, mode, reference):
        check_memory(); gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry = dict(block=source['block'], arm=mode, status='warming', warmup_calls=3,
                     resident_baseline=memory(), repeats=[], counts_before=dict(counters))
        report['benchmark'].append(entry)
        common.save(args.output, report)
        for _ in range(3):
            check_memory(); output = execute(dense, mode); del output
        torch.cuda.synchronize()
        entry.update(status='measuring', after_warmup_baseline=memory())
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for index in range(10):
            check_memory(); torch.cuda.reset_peak_memory_stats()
            before, counts_before = memory(), dict(counters)
            started = time.perf_counter(); begin.record()
            output = execute(dense, mode)
            end.record(); end.synchronize()
            elapsed = (time.perf_counter()-started)*1000
            after = memory()
            entry['repeats'].append(dict(index=index, cuda_ms=begin.elapsed_time(end),
                synchronized_wall_ms=elapsed, before=before, after=after,
                incremental_peak_allocated=after['peak_allocated']-before['allocated'],
                incremental_peak_reserved=after['peak_reserved']-before['reserved'],
                actual_counts={k: counters[k]-counts_before[k] for k in counters}))
            common.save(args.output, report)
            if index != 9:
                del output
        cpu = output.cpu(); del output
        entry['output'] = output_record(cpu, DATA/'benchmark'/f'block{source["block"]}_{mode}.pt')
        entry['versus_bf16'] = numeric(cpu, reference)  # Not a kernel acceptance gate.
        entry['actual_counts'] = {k: counters[k]-entry['counts_before'][k] for k in counters}
        expected = 13*CALLS[mode]
        assert entry['actual_counts']['attention'] == entry['actual_counts']['correction_gemm'] == expected
        for name in ('preprocess', 'k_mean', 'q_pack', 'k_pack', 'v_pack'):
            assert entry['actual_counts'][name] == 13, (mode, name)
        assert entry['actual_counts']['construct'] == (13 if mode == 'codebook' else 0)
        assert entry['actual_counts']['full_correction_gather'] == 0
        assert entry['actual_counts']['q_global_mean'] == (13 if mode in ('global', 'codebook') else 0)
        assert entry['actual_counts']['q_block_mean'] == (0 if mode == 'global' else 13)
        entry.update(status='complete', median_cuda_ms=statistics.median(r['cuda_ms'] for r in entry['repeats']),
            median_synchronized_wall_ms=statistics.median(r['synchronized_wall_ms'] for r in entry['repeats']),
            peak_allocated=max(r['after']['peak_allocated'] for r in entry['repeats']),
            peak_reserved=max(r['after']['peak_reserved'] for r in entry['repeats']),
            max_incremental_peak_allocated=max(r['incremental_peak_allocated'] for r in entry['repeats']),
            max_incremental_peak_reserved=max(r['incremental_peak_reserved'] for r in entry['repeats']))
        common.save(args.output, report)
        print(json.dumps(dict(block=source['block'], arm=mode, cuda_ms=entry['median_cuda_ms'],
                             peak_allocated_gib=entry['peak_allocated']/2**30)), flush=True)

    for source in report['inputs']:
        check_memory()
        raw = common.load(source['capture'])
        dense = tuple(raw[name].transpose(0, 1).contiguous().cuda() for name in ('q', 'k', 'v'))
        reference = common.load(source['outputs']['bf16']['artifact'])
        for mode in ARMS:
            benchmark(source, dense, mode, reference)
        del raw, dense, reference
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    assert report['validation_attention_calls'] == 10 and report['benchmark_attention_calls'] == 468
    assert counters['attention'] == 478
    report['benchmark_counts'] = {k: counters[k]-report['validation_counts'][k] for k in counters}
    assert report['benchmark_counts']['construct'] == 39
    assert report['benchmark_counts']['private_attention'] == 39
    assert report['benchmark_counts']['official_attention'] == 429
    report.update(status='complete', cuda_initialized=True)
    check_memory()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('check', 'run'), required=True)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    args.output = args.output or RD/(args.phase+'.json')
    assert not args.output.exists(), 'Preserve prior attempt'
    started = time.time()
    if args.phase == 'run':
        assert args.deadline_unix and 0 < args.deadline_unix-started <= 900
    def budget():
        if args.deadline_unix and time.time() >= args.deadline_unix:
            raise TimeoutError('E033 shared deadline, including CPU binding and JIT')
    report = dict(experiment='E033', phase=args.phase, status='running', actual_dit_calls=0,
        validation_attention_calls=0, benchmark_attention_calls=0, validation=[], benchmark=[],
        geometry=dict(heads=H, valid_length=N, padded_length=NP, head_dim=D, query_groups=G),
        arms=list(ARMS), warmups=3, repeats=10, expected_validation_calls=10, expected_benchmark_calls=468,
        timing_scope='Resident contiguous BF16 HND QKV to contiguous valid NHD output. Required BF16 means/centering/padding, one Q/K/V pack each, K FP32 conversion, correction GEMMs or fresh E032 codebook construction and T16, native attention, chunk copies/assembly included. No precomputed centers in benchmark. Uploads, CPU metrics, hashes/files, and JIT excluded from timing but inside deadline.',
        memory_scope='Arm-local cache clear with resident QKV; record allocated/reserved before and after warmup, then baseline/peak/increment per repeat. Products released before next repeat. No CUDA graph.',
        validation_policy='Same packets/table native private versus old lookup; finite BF16, signed zeros merged, max representable-step distance <=1. All ten fixed controls complete before the benchmark decision. Violations stop_before_benchmark, preserve artifacts, and are engineering diagnostics, not model-quality scientific negatives.',
        scope='One private single-row TMA consumer prototype and five full resident-path baselines on three saved cases. No deployment, general-backend, algorithm-novelty, or video-quality claim.')
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase == 'check':
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        binding(report, budget)
        if args.phase == 'check':
            assert not torch.cuda.is_initialized()
            report.update(status='complete', cuda_initialized=False)
        else:
            checked = json.loads((RD/'check.json').read_text())
            assert checked['status'] == 'complete' and checked['cuda_initialized'] is False
            for name in ('sources', 'reference_report', 'private_source', 'inputs', 'chunks'):
                assert checked[name] == report[name], name
            with torch.inference_mode():
                run(args, report, budget)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.time()-started
        common.save(args.output, report)
        print(json.dumps(dict(status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
