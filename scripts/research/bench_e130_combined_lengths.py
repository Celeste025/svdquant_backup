#!/usr/bin/env python3
"""Measure a fixed Sage3 build on H3 QKV prefixes; run builds in separate processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt')


def measure(fn, prepare):
    for _ in range(25):
        prepare()
        fn()
    torch.cuda.synchronize()
    batches = []
    for _ in range(6):
        pairs = []
        for _ in range(12):
            prepare()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        batches.append(statistics.mean(a.elapsed_time(b) for a, b in pairs))
    return {'mean_ms': statistics.mean(batches), 'batch_means_ms': batches,
            'cv_batch_percent': 100 * statistics.stdev(batches) / statistics.mean(batches),
            'calls': 72}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sage3-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('baseline', 'combined'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=(1024, 2048, 4096, 8192, 9482, 16384, 22677))
    args = parser.parse_args()
    if any(n < 1 or n > 22677 for n in args.lengths):
        raise ValueError('length outside captured H3 QKV')
    source_dir = args.sage3_dir.resolve()
    for stem in ('fp4attn_cuda', 'fp4quant_cuda'):
        if not list(source_dir.glob(f'{stem}*.so')):
            raise FileNotFoundError(f'{stem} missing from {source_dir}')
    sys.path.insert(0, str(source_dir))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api
    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    if tuple(payload['q'].shape) != (1, 56, 22677, 128):
        raise RuntimeError('unexpected capture shape')
    report = {
        'format': 'e130-combined-lengths-v1',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'capture': str(CAPTURE), 'sage3_dir': str(source_dir), 'mode': args.mode,
        'gpu': torch.cuda.get_device_name(0),
        'attention_extension': str(Path(api.fp4attn_cuda.__file__).resolve()),
        'quant_extension': str(Path(api.fp4quant_cuda.__file__).resolve()),
        'timing_policy': 'CUDA events; 25 warmups + 6x12 calls; K restore before each event; H3 QKV prefixes; online API includes preprocessing and core',
        'rows': [],
    }
    for n in args.lengths:
        q, k, v = (payload[name][:, :, :n, :].to('cuda:0') for name in ('q', 'k', 'v'))
        k_work = torch.empty_like(k)

        def prepare():
            k_work.copy_(k)

        def run():
            if args.mode == 'baseline':
                return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                            per_block_mean=True)
            return sageattn3_blackwell(
                q, k_work, v, is_causal=False, per_block_mean=True,
                fast_preprocess=True, fast_qpack=True,
                fast_delta_gemm=True, fast_raw_k_delta=True)

        with torch.inference_mode():
            prepare()
            out = run()
            torch.cuda.synchronize()
            if not torch.isfinite(out).all():
                raise RuntimeError(f'nonfinite output at n={n}')
            if args.mode == 'combined' and not torch.equal(k_work, k):
                raise RuntimeError(f'input K mutated after call at n={n}')
            digest = hashlib.sha256(out.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            timing = measure(run, prepare)
        row = {'tokens': n, 'shape_hnd': list(q.shape), 'output_sha256': digest,
               'timing': timing}
        report['rows'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'mode': args.mode, 'tokens': n,
                          'mean_ms': timing['mean_ms'], 'sha256': digest}), flush=True)


if __name__ == '__main__':
    main()
