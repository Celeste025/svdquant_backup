#!/usr/bin/env python3
"""Paired online/core H3 Sage2++ versus fully opt-in Sage3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

from bench_e130_combined_lengths import CAPTURE, measure


SAGE2 = Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sage3-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=(8192, 9482, 16384, 22677))
    args = parser.parse_args()
    if any(n < 1 or n > 22677 for n in args.lengths):
        raise ValueError('length outside H3 capture')
    sys.path[:0] = [str(args.sage3_dir.resolve()), str(SAGE2)]
    from sageattention import sageattn, _qattn_sm89
    from sageattention.quant import per_channel_fp8, per_warp_int8
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    report = {'format': 'e130-optimized-vs-sage2pp-two-scope-v1',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'capture': str(CAPTURE),
              'sage2_dir': str(SAGE2), 'sage3_dir': str(args.sage3_dir.resolve()),
              'gpu': torch.cuda.get_device_name(0),
              'scope': 'Online includes each implementation native dynamic QKV quantization/centering/correction, core uses prepared quantized tensors; K restore excluded from timing',
              'flops': '4 * batch * heads * query_tokens * key_tokens * head_dim (noncausal QK + PV, multiply-add=2 FLOPs)',
              'timing': 'CUDA events, 25 warmups + 6x12 calls per arm, forward and reverse arm order',
              'rows': []}
    for n in args.lengths:
        q, k, v = (payload[name][:, :, :n, :].to('cuda:0')
                   for name in ('q', 'k', 'v'))
        b, h, _, d = q.shape
        k_work = torch.empty_like(k)
        scale = d ** -0.5

        def restore_k():
            k_work.copy_(k)

        def sage2_online():
            return sageattn(q, k, v, tensor_layout='HND',
                            is_causal=False, sm_scale=scale)

        def sage3_online():
            return sageattn3_blackwell(
                q, k_work, v, is_causal=False, per_block_mean=True,
                fast_preprocess=True, fast_qpack=True,
                fast_delta_gemm=True, fast_raw_k_delta=True)

        with torch.inference_mode():
            km = k.mean(dim=2, keepdim=True)
            qi, qs, ki, ks = per_warp_int8(q, k, km, tensor_layout='HND',
                                          BLKQ=128, WARPQ=32, BLKK=64)
            v16, vs16, _ = per_channel_fp8(v, tensor_layout='HND',
                                            scale_max=2.25, smooth_v=False)
            out2 = torch.empty_like(q)

            def sage2_core():
                _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                    qi, ki, v16, out2, qs, ks, vs16, 1, 0, 2, scale, 0)
                return out2

            qpack, kpack, vpack, delta_s = api._e130_fast_preprocess_qkv(
                q, k, v, fast_delta_gemm=True, fast_qpack=True,
                fast_raw_k_delta=True)

            def sage3_core():
                return api.blockscaled_fp4_attn(
                    qpack, kpack, vpack, delta_s, n,
                    is_causal=False, per_block_mean=True, is_bf16=True)[0][:, :, :n, :]

            ref2 = sage2_online()
            got2 = sage2_core()
            restore_k()
            ref3 = sage3_online()
            got3 = sage3_core()
            if not torch.equal(ref3, got3):
                raise RuntimeError(f'Sage3 prepared core differs from online API at n={n}')
            diff2 = got2.float() - ref2.float()
            nmse2 = float(diff2.square().sum() / ref2.float().square().sum())
            if nmse2 > 1e-8:
                raise RuntimeError(f'Sage2++ prepared core differs from online API: n={n}, NMSE={nmse2}')
            arms = {'sage2pp_online': (sage2_online, lambda: None),
                    'sage3_online': (sage3_online, restore_k),
                    'sage2pp_core': (sage2_core, lambda: None),
                    'sage3_core': (sage3_core, lambda: None)}
            timings = {name: [] for name in arms}
            for ordering in (tuple(arms), tuple(reversed(arms))):
                for name in ordering:
                    timings[name].append(measure(*arms[name]))
                    print(json.dumps({'tokens': n, 'arm': name,
                                      'round': len(timings[name]),
                                      'ms': timings[name][-1]['mean_ms']}), flush=True)

        means = {name: statistics.mean(t['mean_ms'] for t in rows)
                 for name, rows in timings.items()}
        flops = 4 * b * h * n * n * d
        report['rows'].append({
            'tokens': n, 'shape_hnd': list(q.shape), 'flops': flops,
            'core_online_consistency': {'sage3_bitwise_equal': True,
                                        'sage2pp_nmse': nmse2},
            'means_ms': means,
            'effective_tops': {name: flops / (t * 1e9) for name, t in means.items()},
            'sage3_over_sage2pp_online': means['sage2pp_online'] / means['sage3_online'],
            'sage3_over_sage2pp_core': means['sage2pp_core'] / means['sage3_core'],
            'rounds': timings,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
