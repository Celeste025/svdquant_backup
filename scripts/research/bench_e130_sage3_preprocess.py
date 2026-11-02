#!/usr/bin/env python3
"""A/B Sage3 implicit K/V padding and fused Q pad+mean on real H3 QKV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]

from bench_h3_sage_real_gpu0 import metrics
from bench_h3_sage_two_scope import measure
from e130_sage3_preprocess import attention, prepack

CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE3 = ROOT / 'third_party/sage3_h3_maskfix'
OUTPUT = ROOT / 'results/research/E130/preprocess_ab_gpu2.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.path.insert(0, str(SAGE3))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    if payload['cu_seqlens'].tolist() != [0, 9482, 9536]:
        raise RuntimeError('capture sequence length changed')
    n = 9482
    q, k, v = [payload[name][:n].to('cuda:0').transpose(0, 1).unsqueeze(0)
               for name in ('q', 'k', 'v')]
    if any(x.shape != (1, 56, n, 128) or x.is_contiguous() for x in (q, k, v)):
        raise RuntimeError('not H3 noncontiguous HND QKV')
    k_work = torch.empty_like(k)
    variants = {
        'official': None,
        'fused_q_pad_mean': (True, False, False, False),
        'implicit_kv_pad': (False, True, False, False),
        'fused_q_and_implicit_kv_pad': (True, True, False, False),
        'fused_q_k_center_and_implicit_kv_pad': (True, True, True, False),
        'fused_q_implicit_kv_exact_delta': (True, True, False, True),
    }

    with torch.inference_mode():
        k_work.copy_(k)
        official = sageattn3_blackwell(q, k_work, v, is_causal=False,
                                       per_block_mean=True)
        k_work.copy_(k)
        qo, ko, vo, dso = api.preprocess_qkv(q, k_work, v, True)
        qo_pack = api.scale_and_quant_fp4(qo)
        ko_pack = api.scale_and_quant_fp4_permute(ko)
        vo_pack = api.scale_and_quant_fp4_transpose(vo)
        results = {}
        for name, flags in variants.items():
            if flags is None:
                results[name] = {'vs_official_nmse': 0.0, 'intermediates_equal': True}
                continue
            k_work.copy_(k)
            qpack, kpack, vpack, ds, (qp, _) = prepack(
                q, k_work, v, api, fuse_q_pad_mean=flags[0],
                implicit_kv_pad=flags[1], fuse_k_center_pack=flags[2],
                exact_delta=flags[3],
            )
            checks = {
                'q_center_equal': torch.equal(qp, qo),
                'k_center_equal': torch.equal(k_work, ko[:, :, :n, :]),
                'q_code_equal': torch.equal(qpack[0], qo_pack[0]),
                'q_scale_equal': torch.equal(qpack[1], qo_pack[1]),
                'k_code_equal': torch.equal(kpack[0], ko_pack[0]),
                'k_scale_equal': torch.equal(kpack[1], ko_pack[1]),
                'v_code_equal': torch.equal(vpack[0], vo_pack[0]),
                'v_scale_equal': torch.equal(vpack[1], vo_pack[1]),
                'delta_equal': torch.equal(ds, dso),
            }
            q_difference = (qp.float() - qo.float()).abs()
            ds_difference = (ds.float() - dso.float()).abs()
            packed_output = api.blockscaled_fp4_attn(
                qpack, kpack, vpack, ds, n, is_causal=False,
                per_block_mean=True, is_bf16=True)[0][:, :, :n, :].contiguous()
            comparison = metrics(packed_output, official)
            results[name] = {'vs_official': comparison,
                             'intermediates_equal': all(checks.values()),
                             'checks': checks, 'vs_official_nmse': comparison['nmse'],
                             'q_center_max_abs_diff': float(q_difference.max()),
                             'q_center_mismatched_elements': int((q_difference != 0).sum()),
                             'delta_max_abs_diff': float(ds_difference.max()),
                             'delta_mismatched_elements': int((ds_difference != 0).sum()),
                             'drop_in_accurate': comparison['nmse'] <= 1e-7}
            print(name, json.dumps(results[name]), flush=True)

        def call(name):
            flags = variants[name]
            if flags is None:
                return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                           per_block_mean=True)
            return attention(q, k_work, v, api, fuse_q_pad_mean=flags[0],
                             implicit_kv_pad=flags[1], fuse_k_center_pack=flags[2],
                             exact_delta=flags[3],
                             contiguous_output=True)

        timings = {name: [] for name in variants}
        for ordering in (list(variants), list(reversed(variants))):
            for name in ordering:
                result = measure(lambda name=name: call(name),
                                 prepare=lambda: k_work.copy_(k))
                timings[name].append(result)
                print(f'{name}: {result["mean_ms"]:.4f} ms, '
                      f'CV {result["cv_batch_mean_percent"]:.2f}%', flush=True)
        for name in variants:
            means = [row['mean_ms'] for row in timings[name]]
            results[name]['mean_ms'] = statistics.mean(means)
            results[name]['timing_rounds'] = timings[name]
        baseline = results['official']['mean_ms']
        for name in variants:
            results[name]['speedup_vs_official'] = baseline / results[name]['mean_ms']

    report = {'format': 'e130-sage3-preprocess-ab-v1',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'capture': str(CAPTURE), 'sage3_source': str(SAGE3),
              'shape_hnd': [1, 56, n, 128],
              'input_policy': 'real H3 noncontiguous HND; disposable K restored before CUDA start event',
              'variants': results, 'gpu': torch.cuda.get_device_name(),
              'scope': 'full online Sage3 API including Q/K/V preprocess and FP4 core; K restore excluded'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
