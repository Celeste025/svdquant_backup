#!/usr/bin/env python3
"""Emit one labeled full-H3 Sage3 core launch for Nsight Compute."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/sage3_native_mixed"),
                "/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe"]
import fp4attn_cuda
import fp4attn_mixed_cuda
import sage3_mixed_router_cuda
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("official_fp4", "copied_fp4", "native_mixed"),
                        required=True)
    args = parser.parse_args()
    capture = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
                   "h3_p1_s0_b25_480x832_73f_qkv.pt")
    payload = torch.load(capture, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:9482].transpose(0, 1).unsqueeze(0)
               .contiguous().cuda() for name in ("q", "k", "v"))
    with torch.inference_mode():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packed = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1], ds,
                  9482, None, 128 ** -0.5, False, True, True)
        if args.mode == "official_fp4":
            fn = lambda: fp4attn_cuda.fwd(*packed)
        elif args.mode == "copied_fp4":
            fn = lambda: fp4attn_mixed_cuda.fwd(*packed)
        else:
            v8, scale8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
            if int(route.sum().item()) != 106:
                raise RuntimeError("unexpected native FP8 route count")
            fn = lambda: fp4attn_mixed_cuda.fwd_mixed(*packed, v8, scale8, route)
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
        print(f"E114 profiling mode={args.mode}", flush=True)
        torch.cuda.nvtx.range_push(f"e114_{args.mode}_core")
        out = fn()[0]
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all().item():
            raise RuntimeError("non-finite profile output")


if __name__ == "__main__":
    main()
