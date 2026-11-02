#!/usr/bin/env python3
"""E098 follow-up: official online API versus preprocessed native attention core.

The online timing starts after restoring a disposable K buffer. This models an
inference call receiving fresh K without charging a benchmark-only clone/copy.
Q/K/V are passed in H3's transposed HND view, without a pre-timed contiguous()
conversion. Paper-style timing measures only the native attention core.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

from bench_h3_sage_real_gpu0 import metrics


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE2 = Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')
# RTX PRO 5000 baseline: isolated official Sage3 build with persistent grid
# sized to the device's actual SM count. Pass --sage3-dir for historical 170.
SAGE3 = Path('/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe')
OUTPUT = Path('results/research/E098/sage_h3_sm110_two_scope_gpu0.json')
WARMUP = 25
BATCHES = 12
REPEATS = 12


def measure(fn, prepare=lambda: None):
    """Per-call CUDA events; prepare runs before the start event, never inside it."""
    for _ in range(WARMUP):
        prepare()
        fn()
    torch.cuda.synchronize()
    batches = []
    for _ in range(BATCHES):
        events = []
        for _ in range(REPEATS):
            prepare()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        batches.append([start.elapsed_time(end) for start, end in events])
    batch_means = [statistics.mean(batch) for batch in batches]
    mean_ms = statistics.mean(batch_means)
    return dict(mean_ms=mean_ms,
                median_ms=statistics.median(batch_means),
                std_batch_mean_ms=statistics.stdev(batch_means),
                cv_batch_mean_percent=100 * statistics.stdev(batch_means) / mean_ms,
                batch_mean_ms=batch_means,
                min_call_ms=min(min(batch) for batch in batches),
                max_call_ms=max(max(batch) for batch in batches),
                calls=BATCHES * REPEATS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sage3-dir', type=Path, default=SAGE3)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not (args.sage3_dir / 'fp4attn_cuda.cpython-312-x86_64-linux-gnu.so').is_file():
        raise FileNotFoundError(f'Sage3 extension missing: {args.sage3_dir}')
    sys.path[:0] = [str(SAGE2), str(args.sage3_dir)]
    from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda
    from sageattention import _qattn_sm89
    from sageattention.quant import per_channel_fp8, per_warp_int8
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (blockscaled_fp4_attn, preprocess_qkv,
                               scale_and_quant_fp4, scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location='cpu', weights_only=True, mmap=True)
    if payload['q'].ndim == 3:
        assert payload['format'] == 'h3-480p3s-real-attention-v1'
        assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
        n, heads, dim = 9482, 56, 128
        # The short capture stores H3 [N,H,D]; keep its transposed HND view.
        q, k, v = [payload[name][:n].to('cuda:0').transpose(0, 1).unsqueeze(0)
                   for name in ('q', 'k', 'v')]
        input_layout = 'noncontiguous HND view of H3 NHD'
        excludes = '54-token H3 tail, output assembly, QKV projection/norm/RoPE'
    elif payload['q'].ndim == 4:
        q, k, v = [payload[name].to('cuda:0') for name in ('q', 'k', 'v')]
        _, heads, n, dim = q.shape
        input_layout = 'contiguous HND capture of H3 attention helper'
        excludes = 'output assembly, QKV projection/norm/RoPE'
    else:
        raise RuntimeError('unsupported QKV capture rank')
    assert q.shape == k.shape == v.shape == (1, heads, n, dim)
    if n == 9482:
        assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    k_original = k.clone()
    k_work = torch.empty_like(k)
    sm_scale = dim ** -0.5
    ops = 4 * heads * n * n * dim
    rows = {}

    def add(name, scope, fn, ref, prepare=lambda: None, crop=lambda x: x):
        prepare()
        output = crop(fn())
        error = metrics(output, ref)
        timing = measure(fn, prepare)
        rows.setdefault(name, {})[scope] = dict(
            timing=timing, tops=ops / 1e12 / (timing['mean_ms'] / 1000),
            nmse_vs_bf16=error['nmse'])
        print(f'{name} {scope}: {timing["mean_ms"]:.4f} ms, '
              f'{rows[name][scope]["tops"]:.1f} TOPS, '
              f'CV={timing["cv_batch_mean_percent"]:.2f}%', flush=True)
        return output

    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        add('bf16_sdpa', 'online',
            lambda: F.scaled_dot_product_attention(q, k, v, scale=sm_scale), ref)
        rows['bf16_sdpa']['core_only'] = dict(
            same_as_online=True, timing=rows['bf16_sdpa']['online']['timing'],
            tops=rows['bf16_sdpa']['online']['tops'], nmse_vs_bf16=0.0)

        def sage2_api():
            return sageattn_qk_int8_pv_fp8_cuda(
                q, k, v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_warp', sm_scale=sm_scale,
                pv_accum_dtype='fp32+fp32')

        def sage2pp_api():
            return sageattn(q, k, v, tensor_layout='HND',
                            is_causal=False, sm_scale=sm_scale)

        def sage3_api():
            return sageattn3_blackwell(q, k_work, v,
                                       is_causal=False, per_block_mean=True)

        api_outputs = {}
        for name, fn, prep in (
            ('sage2_fp32', sage2_api, lambda: None),
            ('sage2pp', sage2pp_api, lambda: None),
            ('sage3', sage3_api, lambda: k_work.copy_(k)),
        ):
            api_outputs[name] = add(name, 'online', fn, ref, prep)
            # Sage2's official wrapper is non-mutating on this input. A failure
            # here means the no-clone measurement must be redesigned.
            assert torch.equal(k, k_original)

        # Native cores receive the exact tensors produced by each public API's
        # official preprocessors. Their one-time preparation is deliberately
        # excluded, matching the Sage paper's attention-kernel-only scope.
        km = k.mean(dim=2, keepdim=True)
        qi, qs, ki, ks = per_warp_int8(q, k, km, tensor_layout='HND',
                                      BLKQ=128, WARPQ=32, BLKK=64)
        v32, vs32, _ = per_channel_fp8(v, tensor_layout='HND',
                                        scale_max=448.0, smooth_v=False)
        v16, vs16, _ = per_channel_fp8(v, tensor_layout='HND',
                                        scale_max=2.25, smooth_v=False)
        out32 = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        out16 = torch.empty(q.shape, device=q.device, dtype=q.dtype)

        def sage2_core():
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                qi, ki, v32, out32, qs, ks, vs32, 1, 0, 2, sm_scale, 0)
            return out32

        def sage2pp_core():
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                qi, ki, v16, out16, qs, ks, vs16, 1, 0, 2, sm_scale, 0)
            return out16

        k_pre = k.clone()
        q3, k3, v3, delta_s = preprocess_qkv(q, k_pre, v, True)
        qlist = scale_and_quant_fp4(q3)
        klist = scale_and_quant_fp4_permute(k3)
        vlist = scale_and_quant_fp4_transpose(v3)

        def sage3_core():
            return blockscaled_fp4_attn(qlist, klist, vlist, delta_s, n,
                                        is_causal=False, per_block_mean=True,
                                        is_bf16=True)[0]

        for name, fn, crop in (
            ('sage2_fp32', sage2_core, lambda x: x),
            ('sage2pp', sage2pp_core, lambda x: x),
            ('sage3', sage3_core, lambda x: x[:, :, :n, :].contiguous()),
        ):
            replay = crop(fn())
            agreement = metrics(replay, api_outputs[name])
            if agreement['nmse'] > 1e-8:
                raise RuntimeError(f'{name} core disagrees with official API: {agreement}')
            add(name, 'core_only', fn, ref, crop=crop)
            rows[name]['core_only']['nmse_vs_online'] = agreement['nmse']

    report = dict(format='h3-sage-two-scope-v1', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  capture=str(args.capture), gpu=torch.cuda.get_device_name(0),
                  sage3_source=str(args.sage3_dir.resolve()),
                  shape_hnd=[1, heads, n, dim], input_layout=input_layout,
                  operations=ops, operation_convention='4*B*H*N*N*D, MAC=2 ops',
                  timing='per-call CUDA events; mean of 12 batch means x 12 calls; 25 warmups',
                  online_k_policy='Sage3 disposable K restored before every start event; no clone/copy timed',
                  core_policy='official QKV preprocessing executed once outside timing',
                  excludes=excludes,
                  results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
