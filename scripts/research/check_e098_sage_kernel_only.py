#!/usr/bin/env python3
"""Paper-style core-kernel timing on E098's real MiniMax-H3 QKV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from bench_h3_sage_real_gpu0 import metrics, timed


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE2 = Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')
# RTX PRO 5000 baseline; use --sage3-dir to replay the original 170-block build.
SAGE3 = Path('/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe')
OUTPUT = Path('results/research/E098/sage_h3_sm110_kernel_only_gpu0.json')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-reference-shape', action='store_true',
                        help='Use the official Sage2 benchmark default B4 H32 N8192 D128')
    parser.add_argument('--sage3-dir', type=Path, default=SAGE3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not (args.sage3_dir / 'fp4attn_cuda.cpython-312-x86_64-linux-gnu.so').is_file():
        raise FileNotFoundError(f'Sage3 extension missing: {args.sage3_dir}')
    sys.path[:0] = [str(SAGE2), str(args.sage3_dir)]
    from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda
    from sageattention import _qattn_sm89
    from sageattention.quant import per_channel_fp8, per_warp_int8
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (blockscaled_fp4_attn, preprocess_qkv,
                               scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(0)
    device = 'cuda:0'
    n, heads, dim = (8192, 32, 128) if args.synthetic_reference_shape else (9482, 56, 128)
    batch = 4 if args.synthetic_reference_shape else 1
    with torch.inference_mode():
        if args.synthetic_reference_shape:
            torch.manual_seed(9814)
            q, k, v = [torch.randn((batch, heads, n, dim), device=device,
                                   dtype=torch.bfloat16) for _ in range(3)]
        else:
            payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
            assert payload['format'] == 'h3-480p3s-real-attention-v1'
            assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
            q, k, v = [payload[name][:n].to(device).transpose(0, 1).unsqueeze(0).contiguous()
                       for name in ('q', 'k', 'v')]
        assert q.shape == k.shape == v.shape == (batch, heads, n, dim)
        refs = {
            'sage2_fp32': sageattn_qk_int8_pv_fp8_cuda(
                q, k.clone(), v, tensor_layout='HND', is_causal=False,
                qk_quant_gran='per_warp', sm_scale=dim ** -0.5,
                pv_accum_dtype='fp32+fp32'),
            'sage2pp': sageattn(q, k.clone(), v, tensor_layout='HND',
                               is_causal=False, sm_scale=dim ** -0.5),
            'sage3': sageattn3_blackwell(q, k.clone(), v, is_causal=False,
                                        per_block_mean=True),
        }

        km = k.mean(dim=2, keepdim=True)
        qi, qs, ki, ks = per_warp_int8(q, k, km, tensor_layout='HND',
                                      BLKQ=128, WARPQ=32, BLKK=64)
        # fp32+fp32 uses E4M3 max 448; fp32+fp16 uses the 2.25 scaling
        # selected by SageAttention's public wrapper.
        v32, vs32, _ = per_channel_fp8(v, tensor_layout='HND',
                                        scale_max=448.0, smooth_v=False)
        v16, vs16, _ = per_channel_fp8(v, tensor_layout='HND',
                                        scale_max=2.25, smooth_v=False)
        out32, out16 = torch.empty_like(q), torch.empty_like(q)

        def sage2_core():
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                qi, ki, v32, out32, qs, ks, vs32, 1, 0, 2, dim ** -0.5, 0)
            return out32

        def sage2pp_core():
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                qi, ki, v16, out16, qs, ks, vs16, 1, 0, 2, dim ** -0.5, 0)
            return out16

        q3, k3, v3, delta_s = preprocess_qkv(q, k.clone(), v, True)
        qlist = scale_and_quant_fp4(q3)
        klist = scale_and_quant_fp4_permute(k3)
        vlist = scale_and_quant_fp4_transpose(v3)

        def sage3_core():
            return blockscaled_fp4_attn(qlist, klist, vlist, delta_s, n,
                                        is_causal=False, per_block_mean=True,
                                        is_bf16=True)[0]

        arms = [('sage2_fp32', sage2_core), ('sage2pp', sage2pp_core),
                ('sage3', sage3_core)]
        rows = {}
        ops = 4 * batch * heads * n * n * dim
        for name, fn in arms:
            output = fn()
            if name == 'sage3':
                output = output[:, :, :n, :].contiguous()
            check = metrics(output, refs[name])
            if check['nmse'] > 1e-8:
                raise RuntimeError(f'{name} direct kernel disagrees with public API: {check}')
            result = timed(fn, warmup=30, repeats=20, batches=11)
            result['tops'] = ops / 1e12 / (result['median_ms'] / 1000)
            result['public_api_replay'] = check
            rows[name] = result
            print(f'{name}: {result["median_ms"]:.4f} ms, '
                  f'{result["tops"]:.1f} TOPS, CV={result["cv_percent"]:.2f}%', flush=True)
        output_path = (args.output or
                       (OUTPUT.with_name('sage_sm110_synthetic_b4h32n8192_kernel_only_gpu0.json')
                        if args.synthetic_reference_shape else OUTPUT))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(dict(
            format='e098-paper-style-kernel-only-v1',
            sample='synthetic-normal' if args.synthetic_reference_shape else str(CAPTURE),
            sage3_source=str(args.sage3_dir.resolve()),
            gpu=torch.cuda.get_device_name(0), shape=[batch, heads, n, dim],
            operation_convention='4*B*H*N*N*D, MAC=2 ops',
            operations=ops, warmup=30, repeats=20, batches=11,
            results=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
