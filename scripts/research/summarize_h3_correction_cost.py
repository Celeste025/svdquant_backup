#!/usr/bin/env python3
"""Independent CPU reduction of the fixed E027 cost experiment; no GPU calls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E027'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def record(path):
    p = Path(path).resolve()
    return dict(file=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify(r):
    assert record(r['file']) == r, r['file']
    return Path(r['file'])


def stats(xs):
    assert len(xs) == 10 and all(math.isfinite(x) and x > 0 for x in xs)
    return dict(median=statistics.median(xs), min=min(xs), max=max(xs), repeats=xs)


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def output_error(t, reference):
    assert t.shape == reference.shape == (22539, 56, 128)
    assert t.dtype == reference.dtype == torch.bfloat16
    energy = torch.zeros(56, dtype=torch.float64)
    errors = torch.zeros_like(energy)
    maxima = torch.zeros_like(energy)
    different = torch.zeros(56, dtype=torch.int64)
    for start in range(0, len(t), 256):
        x, y = t[start:start + 256].double(), reference[start:start + 256].double()
        assert x.isfinite().all() and y.isfinite().all()
        delta = x - y
        energy += y.square().sum((0, 2))
        errors += delta.square().sum((0, 2))
        maxima = torch.maximum(maxima, delta.abs().amax((0, 2)))
        different += (x != y).sum((0, 2))
    return dict(pooled_nmse=errors.sum().item() / energy.sum().item(),
                error_energy=errors.sum().item(), reference_energy=energy.sum().item(),
                max_abs=maxima.max().item(), changed_elements=different.sum().item(),
                per_head=[dict(head=h, nmse=(errors[h] / energy[h]).item(),
                               error_energy=errors[h].item(), reference_energy=energy[h].item(),
                               max_abs=maxima[h].item(), changed_elements=different[h].item())
                          for h in range(56)])


def summarize(report):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    documents = {}
    for name in ('chunk_run', 'stripe_run', 'chunk_launcher', 'stripe_launcher', 'chunk_check', 'stripe_check'):
        path = RD / (name + '.json')
        report['inputs'][name] = record(path)
        documents[name] = json.loads(path.read_text())
        assert documents[name]['status'] == 'complete', name
    chunk, stripe = documents['chunk_run'], documents['stripe_run']
    for kind in ('chunk', 'stripe'):
        launcher = documents[kind + '_launcher']
        assert launcher['returncode'] == 0 and launcher['gpu'] == 0
        assert launcher['result_sha256'] == report['inputs'][kind + '_run']['sha256']
        verify(launcher['source']); verify(launcher['cpu_check'])
    assert documents['chunk_launcher']['gpu_before']['uuid'] == documents['stripe_launcher']['gpu_before']['uuid']
    assert documents['stripe_launcher']['start_epoch'] + documents['stripe_launcher']['seconds'] < documents['chunk_launcher']['start_epoch']
    assert not documents['chunk_check']['cuda_initialized'] and not documents['stripe_check']['cuda_initialized']
    assert chunk['complete_dit_calls'] == stripe['actual_dit_calls'] == stripe['actual_attention_calls'] == 0
    assert all(p['exact'] for p in chunk['packet_parity'].values())
    report.update(geometry=chunk['geometry'], environment=chunk['environment'],
                  timing_scope=chunk['timing_scope'], memory_scope=chunk['memory_scope'],
                  source_case=chunk['source_case'], arms={})
    report['reference'] = chunk['inputs']['output']
    reference = torch.load(verify(report['reference']), map_location='cpu', weights_only=False, mmap=True)
    report['reference_tensor_sha256'] = tensor_sha(reference)
    totals = {key: 0 for key in chunk['actual_total_counts']}
    for name, arm in chunk['arms'].items():
        assert arm['status'] == 'complete' and arm['warmup_calls'] == 3
        raw = arm['repeats']
        assert [r['index'] for r in raw] == list(range(10))
        timing = {key: stats([r[key] for r in raw]) for key in ('cuda_ms', 'synchronized_wall_ms')}
        for key in timing:
            assert timing[key]['median'] == arm['median_' + key]
        counts = arm['actual_counts']
        attention_calls = 13 * (6 if name.endswith('chunk4096') else 1)
        assert counts['attention'] == counts['correction_gemm'] == attention_calls
        for key in ('preprocess', 'k_global_mean', 'q_pack', 'k_pack', 'v_pack'):
            assert counts[key] == (13 if name.startswith('end_to_end/') else 0)
        for key in totals:
            totals[key] += counts[key]
        artifact = arm['output']['artifact']
        value = torch.load(verify(artifact), map_location='cpu', weights_only=False, mmap=True)
        got_sha = tensor_sha(value)
        assert got_sha == arm['output']['tensor']['sha256']
        numerical = output_error(value, reference)
        assert math.isclose(numerical['pooled_nmse'], arm['versus_E026_original_block']['pooled_nmse'], rel_tol=1e-12, abs_tol=1e-18)
        assert numerical['max_abs'] == arm['versus_E026_original_block']['max_abs']
        del value
        peak_a = max(r['after']['peak_allocated'] for r in raw)
        peak_r = max(r['after']['peak_reserved'] for r in raw)
        increments = [r['after']['peak_allocated'] - r['before']['allocated'] for r in raw]
        assert increments == [r['incremental_peak_allocated'] for r in raw]
        assert (peak_a, peak_r, max(increments)) == (arm['peak_allocated'], arm['peak_reserved'], arm['max_incremental_peak_allocated'])
        report['arms'][name] = dict(timing=timing, counts=counts, artifact=artifact,
            tensor_sha256=got_sha, tensor_bytes_equal_reference=got_sha == report['reference_tensor_sha256'],
            versus_E026_original_block=numerical, memory_bytes=dict(
                resident_baseline=arm['resident_baseline'], after_warmup_baseline=arm['after_warmup_baseline'],
                repeat_baseline_allocated=sorted({r['before']['allocated'] for r in raw}),
                repeat_baseline_reserved=sorted({r['before']['reserved'] for r in raw}),
                peak_allocated=peak_a, peak_reserved=peak_r, max_incremental_peak_allocated=max(increments)))
    for key in ('preprocess', 'k_global_mean', 'q_pack', 'k_pack', 'v_pack'):
        totals[key] += 1  # Explicit startup packet-parity preparation, outside timed arms.
    assert totals == chunk['actual_total_counts'] and totals['attention'] == 182
    report['actual_counts'] = dict(chunk=totals, stripe_launches=stripe['stripe_launches'], complete_dit_calls=0)
    report['chunk_relative_to_full'] = {}
    for tier in ('packed_consumer', 'end_to_end'):
        full, partial = [report['arms'][tier + '/' + suffix] for suffix in ('full', 'chunk4096')]
        report['chunk_relative_to_full'][tier] = {
            'cuda_time_change_percent': 100 * (partial['timing']['cuda_ms']['median'] / full['timing']['cuda_ms']['median'] - 1),
            'wall_time_change_percent': 100 * (partial['timing']['synchronized_wall_ms']['median'] / full['timing']['synchronized_wall_ms']['median'] - 1),
            **{key + '_reduction_percent': 100 * (1 - partial['memory_bytes'][key] / full['memory_bytes'][key])
               for key in ('peak_allocated', 'peak_reserved', 'max_incremental_peak_allocated')}}
    stripe_timing = stats(stripe['timing_ms']['repeats'])
    assert stripe_timing['median'] == stripe['timing_ms']['median'] and stripe['stripe_launches'] == 13
    report['stripe'] = dict(timing_cuda_ms=stripe_timing, scope=stripe['scope'],
        actual_launches=13, warmup_calls=3, attention_calls=0, dit_calls=0,
        numeric_check_reported_only=stripe['numeric_check'], compiled_resources=stripe['compiled_resources'],
        peak_allocated_gib_fixture_inclusive=stripe['peak_allocated_gib'],
        memory_caveat='No peak reset: lifetime peak includes original GPU Q/K/V, official full FP32 correction reference, BF16 Kc/mu, stripe output and expected/check tensors. Not fused workspace or incremental memory.',
        interpretation='Standalone 16-row BF16 MMA traversal with 15 repeated unused rows, checksum stores, independent CTA/cache behavior; neither a fused latency nor lower bound. Do not add/subtract it from whole-attention timings.')
    report['correction_storage'] = dict(**chunk['correction_storage'],
        reduction_percent=100 * (1 - chunk['correction_storage']['max_chunk_bytes'] / chunk['correction_storage']['full_bytes']))
    failure = RD / 'cpu_float8_preflight1.json'
    report['preflight_history'] = dict(report=record(failure), details=json.loads(failure.read_text()),
        archived_source=record(RD / 'bench_h3_chunked_correction_preflight1.py'),
        archived_check=record(RD / 'chunk_check_preflight1.json'), final_check=report['inputs']['chunk_check'],
        scope='CPU Float8 isfinite validation failure fixed before any E027 GPU run. Raw-byte hashing and measured operator semantics preserved; final CPU check complete. No hidden GPU retry.')
    report['limitations'] = [
        'One fixed real p36/step14/block0 input and one GPU; 10 repeats quantify this invocation, not independent model/data samples.',
        'Four saved last-repeat outputs independently reduced in CPU FP64; intermediate repeat outputs were not saved.',
        'Reported total allocated peaks include tier-resident operands and runtime allocations; incremental peaks subtract each measured invocation baseline. Reserved memory is allocator-history dependent.',
        'Chunk cost includes six GEMMs, native launches, required packet copies and output assembly; QKV tier adds shared preprocessing once. Upload, validation and CPU saving excluded.',
        'Stripe peak includes reference fixtures; stripe time cannot be algebraically combined with whole-attention time.',
        'Known chunk baseline trades time for memory at this shape; no full-model deployment gain or universal infeasibility of fused kernels is established.']
    assert not torch.cuda.is_initialized()
    report.update(status='complete', cuda_initialized=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=RD / 'cost_summary.json')
    args = p.parse_args()
    assert not args.output.exists(), 'Preserve previous summary'
    result = dict(experiment='E027', status='running', source=record(__file__), inputs={})
    start = time.time()
    try:
        summarize(result)
    except BaseException:
        result.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        result['cpu_seconds'] = time.time() - start
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__':
    import torch
    main()
