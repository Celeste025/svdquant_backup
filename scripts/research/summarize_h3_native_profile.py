#!/usr/bin/env python3
"""CPU-only validation/summary of frozen E009 full resident DiT profiles."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('bf16', 'qdq', 'native')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_trace(profile):
    """Re-attribute strict kernel events independently of saved aggregation."""
    path = Path(profile['trace']['path'])
    assert sha256(path) == profile['trace']['sha256'], 'Trace SHA drift'
    events = json.loads(path.read_text())['traceEvents']
    semantic = defaultdict(list)
    for event in events:
        if event.get('cat') == 'user_annotation' and event.get('name', '').startswith('E009::'):
            if event['name'] != 'E009::complete_forward':
                semantic[event['pid'], event['tid']].append(event)
    indexes = {}
    for key, ranges in semantic.items():
        ranges.sort(key=lambda event: event['ts'])
        for left, right in zip(ranges, ranges[1:]):
            assert left['ts'] + left['dur'] <= right['ts'] + 1e-3, 'Semantic overlap'
        indexes[key] = ([event['ts'] for event in ranges], ranges)
    launches = {}
    for event in events:
        if event.get('cat') in ('cuda_runtime', 'cuda_driver'):
            correlation = event.get('args', {}).get('correlation')
            if correlation is not None:
                assert correlation not in launches, 'Nonunique correlation'
                launches[correlation] = event
    counts, work = Counter(), defaultdict(float)
    kernels = [event for event in events if event.get('cat') == 'kernel']
    for event in kernels:
        assert not event['name'].startswith('E009::'), 'Annotation counted as kernel'
        launch = launches[event['args']['correlation']]
        starts, ranges = indexes.get((launch['pid'], launch['tid']), ([], []))
        index = bisect_right(starts, launch['ts']) - 1
        category = 'other'
        if index >= 0 and launch['ts'] < ranges[index]['ts'] + ranges[index]['dur']:
            category = ranges[index]['name']
        counts[category] += 1
        work[category] += event['dur']
    total = sum(event['dur'] for event in kernels)
    assert len(kernels) == profile['strict_kernel_count']
    assert abs(total - profile['strict_kernel_work_us']) < 1e-4
    assert set(counts) == {row['category'] for row in profile['categories']}
    for row in profile['categories']:
        name = row['category']
        assert counts[name] == row['kernel_calls']
        assert abs(work[name] - row['kernel_work_us']) < 1e-4
    assert abs(sum(work.values()) - total) < 1e-4
    d2h = [e for e in events if e.get('cat') == 'gpu_memcpy' and 'DtoH' in e['name']]
    d2h_categories = Counter()
    d2h_api_us = 0.
    for event in d2h:
        launch = launches[event['args']['correlation']]
        d2h_api_us += launch['dur']
        starts, ranges = indexes.get((launch['pid'], launch['tid']), ([], []))
        index = bisect_right(starts, launch['ts']) - 1
        category = 'other'
        if index >= 0 and launch['ts'] < ranges[index]['ts'] + ranges[index]['dur']:
            category = ranges[index]['name']
        d2h_categories[category] += 1
    return {'trace_sha_verified': True, 'unique_launch_correlations': True,
            'semantic_ranges_nonoverlapping': True, 'classification_recomputed_exact': True,
            'kernel_count': len(kernels), 'kernel_work_us': total,
            'excluded_gpu_user_annotations': sum(e.get('cat') == 'gpu_user_annotation' for e in events),
            'd2h_sync_audit': {
                'calls': len(d2h), 'bytes_histogram': dict(Counter(e['args']['bytes'] for e in d2h)),
                'semantic_categories': dict(d2h_categories),
                'gpu_copy_work_us': sum(e['dur'] for e in d2h), 'cpu_copy_api_us': d2h_api_us,
                'item_or_local_scalar_dense_calls': sum(e.get('cat') == 'cpu_op' and e['name'] in
                                                       ('aten::item', 'aten::_local_scalar_dense') for e in events),
                'interpretation': '52 copies of 12 bytes outside native GEMM match H3 cu_seqlens.tolist() metadata reads (2 refiner plus 50 main attention modules). Long CPU memcpy duration includes waiting for earlier queued GPU work; not transfer bandwidth cost. No evidence for 200 TensorWise global scalar reads. CUDA graph support is not established.'}}


def verify_checks(row):
    assert row['checked_calls'] == 200 and row['invalid_calls'] == 0
    assert row['calls_with_nonzero_input_zero_sf'] == 1
    assert row['affected_call_indices_zero_based'] == [7]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E009_profile_summary.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    output = {
        'experiment': 'E009_h3_resident_profile_summary', 'status': 'complete',
        'source': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__))},
        'scope': 'One fixed p1 step0 calibration-cache full DiT call; 22400 packed tokens; 50 main blocks, 200 native target linears, 8 BF16 refiner linears. Sequential independent processes on GPU5, one warmup, three unprofiled repetitions and one independent profiler per arm.',
        'baseline': 'All weights resident, same eager torch and BF16 SDPA dispatch. QDQ residual weights decoded once at startup; original common activation/smooth/LR hooks in timed forwards. Native uses unfused torch scaled_mm with same BF16 smooth/LR and checked fast packing.',
        'limitations': [
            'This measures one full denoiser call, not video generation latency, serving throughput, held-out quality or a research contribution.',
            'Endpoint SHA matches each arm own frozen reference; native differs numerically from old QDQ/BF16, so this is not a quality-equivalent acceleration claim.',
            'Single sample and three repetitions do not establish multi-shape/device or production performance. BF16 is the current eager implementation, not an optimized ceiling.',
            'Only Chrome cat=kernel counts as kernel work. Work fractions are not critical-path shares; GPU annotations, copies, memsets and blocking CPU API durations must not be added again.',
            'Steady memory excludes startup/load conversion and profiler; startup peak is separate. Reserved memory depends on allocator history. Unique resident storage includes hook-owned LR/smooth, not CUDA library state.',
            'Native SF0 checks report affected calls, not groups; all repeats have one affected call at index 7. The separate real-input proof counted 95 affected audio groups.'
        ],
        'arms': {},
    }
    common_files = None
    for arm in ARMS:
        path = ROOT/f'results/research/E009_profile_{arm}.json'
        raw = json.loads(path.read_text())
        assert raw['status'] == 'complete' and raw['arm'] == arm
        if common_files is None:
            common_files = raw['files']
        else:
            assert raw['files'] == common_files, 'Different provenance across arms'
        expected = raw['expected_endpoint_sha256']
        assert raw['warmup']['endpoints_sha256'] == expected
        assert len(raw['unprofiled_repeats']) == 3
        assert all(row['endpoints_sha256'] == expected for row in raw['unprofiled_repeats'])
        profile = raw['profile']
        assert profile['endpoint_sha256'] == expected
        assert profile['counts']['attention'] == 102
        if arm == 'native':
            for row in [raw['warmup']['timing_excluded_from_steady'], *raw['unprofiled_repeats']]:
                assert row['fastpack_checks'] == 200
                verify_checks(row['zero_sf_check_summary'])
            verify_checks(profile['zero_sf_check_summary'])
            assert profile['counts']['native_gemm'] == 200
            assert sum(row['calls'] for row in profile['kernel_aggregate']
                       if 'sm120' in row['name'] and 'e2m1' in row['name']) == 200
        assert sha256(raw['snapshot']['path']) == raw['snapshot']['sha256']
        row = {'raw_path': str(path), 'raw_sha256': sha256(path),
               'warmup_three_repeats_profile_endpoint_sha_exact': True,
               'trace_verification': verify_trace(profile)}
        for key in ('latency_ms', 'steady_before', 'steady_memory', 'resident_model_storage',
                    'startup_seconds', 'startup_peak_allocated_bytes', 'expected_endpoint_sha256',
                    'correctness_binding', 'snapshot', 'device', 'torch', 'cuda', 'sdpa_enabled'):
            row[key] = raw[key]
        row['memory_gib'] = {
            'resident_model_storage': raw['resident_model_storage']['unique_cuda_storage_bytes']/2**30,
            'steady_peak_allocated': raw['steady_memory']['peak_allocated_bytes']/2**30,
            'steady_peak_reserved': raw['steady_memory']['peak_reserved_bytes']/2**30,
            'startup_peak_allocated': raw['startup_peak_allocated_bytes']/2**30}
        row['profile'] = {k: v for k, v in profile.items() if k != 'categories'}
        row['profile']['categories'] = [{k: v for k, v in group.items() if k != 'kernels'}
                                       for group in profile['categories']]
        output['arms'][arm] = row
    median = lambda arm: output['arms'][arm]['latency_ms']['host_ms_including_checks']['median']
    output['matched_single_dit_latency_ratios'] = {
        'bf16_over_native': median('bf16')/median('native'),
        'qdq_over_native': median('qdq')/median('native'),
        'native_latency_reduction_fraction_vs_bf16': 1-median('native')/median('bf16'),
        'metric': 'Unprofiled median host wall, including flag checks; separate serial processes on GPU5'}
    output['memory_ratios'] = {
        'native_over_bf16_resident': output['arms']['native']['memory_gib']['resident_model_storage']/output['arms']['bf16']['memory_gib']['resident_model_storage'],
        'native_over_bf16_steady_peak_allocated': output['arms']['native']['memory_gib']['steady_peak_allocated']/output['arms']['bf16']['memory_gib']['steady_peak_allocated']}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    for arm, row in output['arms'].items():
        print(arm, 'host median ms', median(arm), 'memory GiB', row['memory_gib'])
        for group in row['profile']['categories']:
            print(' ', group['category'], round(group['kernel_work_us']/1000, 3), 'ms',
                  round(group['fraction_of_kernel_work']*100, 3), '%')
    print(output['matched_single_dit_latency_ratios'])
    print(args.output)


if __name__ == '__main__':
    main()
