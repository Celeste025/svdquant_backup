#!/usr/bin/env python3
"""Emit one H3 mixed-core launch for a selected Nsight Compute profile."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("baseline", "inplace"), required=True)
    parser.add_argument("--route", choices=("zero", "real"), default="zero")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed")]
    import fp4attn_mixed_maskfix_cuda
    import fp4attn_mixed_inplace_cuda
    import sage3_mixed_router_sparse_cuda
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:9482].to("cuda:0").transpose(0, 1).unsqueeze(0)
               for name in ("q", "k", "v"))
    with torch.inference_mode():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, 9482, None, 128 ** -0.5, False, True, True)
        v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(vp, 0.10)
        if args.route == "zero":
            route = torch.zeros_like(route)
        core = (fp4attn_mixed_maskfix_cuda if args.candidate == "baseline"
                else fp4attn_mixed_inplace_cuda)

        def launch():
            return core.fwd_mixed(*packet, v8, s8, route)[0]

        for _ in range(8):
            launch()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"e118_{args.candidate}_{args.route}")
        out = launch()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("non-finite core output")
        print(f"profiled {args.candidate} {args.route}", flush=True)


if __name__ == "__main__":
    main()
