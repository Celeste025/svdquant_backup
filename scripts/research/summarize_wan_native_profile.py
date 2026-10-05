#!/usr/bin/env python3
"""CPU-only E007 trace correction; never edit or replace frozen raw results.

Torch 2.11 key_averages includes both CPU and GPU user annotations. Those
annotations are not kernels. This script counts only Chrome cat=kernel,
attributes each kernel through its unique launch correlation to disjoint CPU
semantic ranges, and leaves unlabelled work as other. No inference is rerun.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path('/data1/models/svdquant-wjq/research/20261002/E007/profile')
ARMS = ('bf16', 'qdq', 'native')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def aggregate(events):
    groups = defaultdict(list)
    for event in events:
        groups[event['name']].append(float(event.get('dur', 0)))
    return sorted([
        {'name': name, 'calls': len(values), 'total_us': sum(values),
         'mean_us': sum(values)/len(values), 'min_us': min(values), 'max_us': max(values)}
        for name, values in groups.items()
    ], key=lambda row: row['total_us'], reverse=True)


def analyze_trace(path):
    trace = json.loads(path.read_text())
    events = trace['traceEvents']
    cpu_annotations = [e for e in events if e.get('cat') == 'user_annotation'
                       and e.get('name', '').startswith('E007::')]
    semantic = [e for e in cpu_annotations if e['name'] != 'E007::complete_dit_forward']
    by_thread = defaultdict(list)
    for event in semantic:
        by_thread[(event['pid'], event['tid'])].append(event)
    indexes = {}
    for key, ranges in by_thread.items():
        ranges.sort(key=lambda event: event['ts'])
        for left, right in zip(ranges, ranges[1:]):
            if left['ts'] + left['dur'] > right['ts'] + 1e-3:
                raise RuntimeError(f'CPU semantic ranges overlap: {left["name"]}, {right["name"]}')
        indexes[key] = ([event['ts'] for event in ranges], ranges)
    launch = {}
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    for event in runtime:
        correlation = event.get('args', {}).get('correlation')
        if correlation is None:
            continue
        if correlation in launch:
            raise RuntimeError(f'Nonunique CPU launch correlation {correlation}')
        launch[correlation] = event

    kernel_groups = defaultdict(list)
    missing_launch = []
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        category = 'other'
        start = launch.get(event.get('args', {}).get('correlation'))
        if start is None:
            missing_launch.append(event['name'])
        else:
            starts, ranges = indexes.get((start['pid'], start['tid']), ([], []))
            index = bisect_right(starts, start['ts']) - 1
            if index >= 0:
                enclosing = ranges[index]
                if start['ts'] < enclosing['ts'] + enclosing['dur']:
                    category = enclosing['name'].removeprefix('E007::')
        kernel_groups[category].append(event)
    if missing_launch:
        raise RuntimeError(f'{len(missing_launch)} kernels have no launch correlation')
    kernels = [e for e in events if e.get('cat') == 'kernel']
    total_us = sum(e['dur'] for e in kernels)
    semantic_rows = []
    for name, members in kernel_groups.items():
        work = sum(e['dur'] for e in members)
        semantic_rows.append({'category': name, 'kernel_calls': len(members),
                              'kernel_work_us': work, 'fraction_of_kernel_work': work/total_us,
                              'kernel_names': aggregate(members)})
    semantic_rows.sort(key=lambda row: row['kernel_work_us'], reverse=True)
    if abs(sum(row['kernel_work_us'] for row in semantic_rows) - total_us) > 1e-5:
        raise RuntimeError('Kernel classification does not conserve total work')
    return {
        'trace_path': str(path), 'trace_sha256': sha256(path),
        'event_category_counts': dict(Counter(e.get('cat', '<none>') for e in events)),
        'cpu_semantic_calls': dict(Counter(e['name'] for e in cpu_annotations)),
        'strict_kernel_count': len(kernels), 'strict_kernel_work_us': total_us,
        'strict_kernel_aggregate': aggregate(kernels),
        'mutually_exclusive_kernel_classification': semantic_rows,
        'missing_launch_correlations': len(missing_launch),
        'copy_events': aggregate([e for e in events if e.get('cat') == 'gpu_memcpy']),
        'memset_events': aggregate([e for e in events if e.get('cat') == 'gpu_memset']),
        'cpu_cuda_api_events': aggregate(runtime),
        'excluded_gpu_user_annotation_count': sum(e.get('cat') == 'gpu_user_annotation' for e in events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E007_profile_summary.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    output = {
        'experiment': 'E007_profile_trace_correction', 'status': 'complete',
        'source': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__))},
        'method': 'Only Chrome cat=kernel durations count as kernel work. Each unique kernel launch correlation maps to a cuda_runtime/cuda_driver CPU event; its CPU timestamp assigns at most one disjoint E007 semantic CPU annotation. Unlabelled kernels remain other. Copies/memsets and GPU annotations are reported separately, never added as kernels.',
        'raw_aggregation_issue': 'Frozen raw annotation_ranges and cuda_kernel_aggregate include GPU user annotations under torch 2.11. These duplicate/nested ranges invalidate their work shares; unprofiled timing, endpoint SHA, memory and raw Chrome traces remain valid. Raw artifacts are unchanged.',
        'limitations': ['One fixed calibration-cache denoiser call, 31200 video tokens; no text encoder, VAE, 4-step rollout or held-out quality claim.',
                        'Three unprofiled repetitions after one warmup; one independent profiler pass per arm. Kernel work is not critical-path latency. Profiler timings are not used for latency speed ratios.',
                        'Native output matches the earlier E007 native endpoint exactly, but differs from old QDQ (E007 NMSE 0.0427664). This is deployment feasibility, not a quality-equivalent acceleration claim.',
                        'Other kernels remain unclassified; in QDQ this includes legacy activation quantization work. Do not call all other work quantization.',
                        'Peak allocated/reserved exclude startup conversion and independent profiler; startup is separately reported. Resident storage includes deduplicated hook LR and smoothing.'],
        'arms': {},
    }
    for arm in ARMS:
        raw_path = ROOT/f'results/research/E007_profile_{arm}.json'
        raw = json.loads(raw_path.read_text())
        if raw['status'] != 'complete':
            raise RuntimeError(f'Arm {arm} is not complete')
        row = {'raw_path': str(raw_path), 'raw_sha256': sha256(raw_path)}
        for key in ('expected_endpoint_sha256', 'latency_ms', 'steady_before', 'steady_memory',
                    'resident_model_storage', 'startup_seconds_including_hash_load_conversion',
                    'startup_peak_allocated_bytes', 'warmup', 'snapshot'):
            row[key] = raw[key]
        row['trace'] = analyze_trace(ARTIFACTS/f'E007_profile_{arm}.trace.json')
        output['arms'][arm] = row
    median = lambda arm: output['arms'][arm]['latency_ms']['host_ms_including_checks']['median']
    output['matched_single_dit_latency_ratios'] = {
        'qdq_over_native': median('qdq')/median('native'),
        'bf16_over_native': median('bf16')/median('native'),
        'native_slowdown_fraction_vs_bf16': median('native')/median('bf16') - 1,
        'metric': 'median host wall including flags validation; separate serial processes on GPU0',
    }
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    for arm, row in output['arms'].items():
        print(arm, 'host_median_ms=', median(arm), 'kernel_total_ms=', row['trace']['strict_kernel_work_us']/1000)
        for group in row['trace']['mutually_exclusive_kernel_classification']:
            print(' ', group['category'], group['kernel_calls'], round(group['kernel_work_us']/1000, 3), 'ms')
    print(json.dumps(output['matched_single_dit_latency_ratios'], indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
