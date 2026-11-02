#!/usr/bin/env python3
"""One labeled native Sage core launch for Nsight Compute on real H3 QKV."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from bench_h3_sage_real_gpu0 import metrics


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE2 = Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')
SAGE3 = Path('/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('sage2pp', 'sage3'), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sage3-dir', type=Path, default=SAGE3)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--tokens', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--heads', type=int, default=56)
    args = parser.parse_args()
    sys.path[:0] = [str(SAGE2), str(args.sage3_dir)]
    from sageattention import sageattn, _qattn_sm89
    from sageattention.quant import per_channel_fp8, per_warp_int8
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose,
                               blockscaled_fp4_attn)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(args.capture, map_location='cpu', weights_only=True, mmap=True)
    if payload['q'].ndim == 3:
        assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
        n = args.tokens or 9482
        if args.batch_size != 1 or args.heads != 56 or n > 9482:
            raise ValueError('rank-3 H3 capture only supports B1H56 and <=9482 tokens')
        q, k, v = [payload[name][:n].to(f'cuda:{args.gpu}').transpose(0, 1).unsqueeze(0)
                   for name in ('q', 'k', 'v')]
    else:
        n = args.tokens or 22677
        if not 1 <= args.heads <= 56 or not 1 <= n <= 22677 or args.batch_size < 1:
            raise ValueError('requested shape exceeds H3 capture')
        q, k, v = [payload[name][:, :args.heads, :n, :].to(f'cuda:{args.gpu}')
                   .expand(args.batch_size, -1, -1, -1).contiguous()
                   for name in ('q', 'k', 'v')]
    assert all(z.shape == (args.batch_size, args.heads, n, 128) for z in (q, k, v))

    with torch.inference_mode():
        if args.arm == 'sage2pp':
            reference = sageattn(q, k, v, tensor_layout='HND',
                                 is_causal=False, sm_scale=128 ** -0.5)
            km = k.mean(dim=2, keepdim=True)
            qi, qs, ki, ks = per_warp_int8(q, k, km, tensor_layout='HND',
                                          BLKQ=128, WARPQ=32, BLKK=64)
            vi, vs, _ = per_channel_fp8(v, tensor_layout='HND',
                                        scale_max=2.25, smooth_v=False)
            output = torch.empty_like(reference)

            def core():
                _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
                    qi, ki, vi, output, qs, ks, vs, 1, 0, 2, 128 ** -0.5, 0)
                return output
        else:
            reference = sageattn3_blackwell(q, k.clone(), v,
                                             is_causal=False, per_block_mean=True)
            qp, kp, vp, delta_s = preprocess_qkv(q, k.clone(), v, True)
            qlist = scale_and_quant_fp4(qp)
            klist = scale_and_quant_fp4_permute(kp)
            vlist = scale_and_quant_fp4_transpose(vp)

            def core():
                return blockscaled_fp4_attn(qlist, klist, vlist, delta_s, n,
                                            is_causal=False, per_block_mean=True,
                                            is_bf16=True)[0]

        for _ in range(8):
            result = core()
        result = result[:, :, :n, :].contiguous() if args.arm == 'sage3' else result
        agreement = metrics(result, reference)
        if agreement['nmse'] != 0.0:
            raise RuntimeError(f'Core does not reproduce public API: {agreement}')
        torch.cuda.synchronize()
        print(f'{args.arm}: core/API NMSE={agreement["nmse"]}; profiling one launch', flush=True)
        torch.cuda.nvtx.range_push('e098_native_core')
        result = core()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(f'{args.arm}: completed', flush=True)


if __name__ == '__main__':
    main()
