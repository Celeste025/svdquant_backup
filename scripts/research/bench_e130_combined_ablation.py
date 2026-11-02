#!/usr/bin/env python3
"""Cumulative Sage3 optimization ablation on captured H3 QKV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

from bench_e130_combined_lengths import CAPTURE, measure


ARMS = (
    ('opt_build_public', {}),
    ('fast_preprocess', {'fast_preprocess': True}),
    ('plus_qpack', {'fast_preprocess': True, 'fast_qpack': True}),
    ('plus_delta_gemm', {'fast_preprocess': True, 'fast_qpack': True,
                         'fast_delta_gemm': True}),
    ('plus_raw_k', {'fast_preprocess': True, 'fast_qpack': True,
                    'fast_delta_gemm': True, 'fast_raw_k_delta': True}),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sage3-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=(8192, 9482, 16384, 22677))
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    report = {'format': 'e130-combined-ablation-v1',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'capture': str(CAPTURE), 'sage3_dir': str(args.sage3_dir.resolve()),
              'gpu': torch.cuda.get_device_name(0),
              'policy': 'Same optimized build; cumulative API flags; CUDA events; 25 warmups and 6x12 calls per arm per order; K restore outside timing',
              'rows': []}
    for n in args.lengths:
        q, k, v = (payload[name][:, :, :n, :].to('cuda:0')
                   for name in ('q', 'k', 'v'))
        k_work = torch.empty_like(k)

        def prepare():
            k_work.copy_(k)

        def call(flags):
            return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                        per_block_mean=True, **flags)

        with torch.inference_mode():
            prepare()
            reference = call(ARMS[0][1])
            equality = {}
            for name, flags in ARMS:
                prepare()
                out = call(flags)
                equality[name] = bool(torch.equal(reference, out))
                if not equality[name]:
                    raise RuntimeError(f'{name} differs from public API at n={n}')
            timings = {name: [] for name, _ in ARMS}
            for order in (ARMS, tuple(reversed(ARMS))):
                for name, flags in order:
                    timings[name].append(measure(lambda flags=flags: call(flags), prepare))
                    print(json.dumps({'tokens': n, 'arm': name,
                                      'round': len(timings[name]),
                                      'ms': timings[name][-1]['mean_ms']}), flush=True)
        means = {name: statistics.mean(t['mean_ms'] for t in rows)
                 for name, rows in timings.items()}
        report['rows'].append({'tokens': n, 'shape_hnd': list(q.shape),
                               'bitwise_equal_public': equality,
                               'means_ms': means, 'timings': timings})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
