#!/usr/bin/env python3
"""Benchmark the full input-to-output E120 split FP8 attention path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--bm", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--warps", type=int, choices=(2, 4, 8, 16), default=4)
    parser.add_argument("--stages", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E120/full_online.json")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed"),
                    str(ROOT / "scripts/research")]
    import fp4attn_lse_cuda
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_sparse_cuda
    from e117_full_api_benchmark import time_cuda
    from e119_sparse_v_correction import compact_route
    from e120_split_native_mixed import (dequant_qk4, split_selected_fp8_pv,
                                         zero_selected_v4)
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:9482].to(f"cuda:{args.gpu}")
               .transpose(0, 1).unsqueeze(0) for name in ("q", "k", "v"))
    k_work = torch.empty_like(k)
    qlen, heads, d = 9482, 56, 128

    def prepare():
        k_work.copy_(k)

    def pack():
        qp, kp, vp, ds = preprocess_qkv(q, k_work, v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, qlen, None, d ** -0.5, False, True, True)
        return qp, kp, vp, q4, k4, v4, ds, packet

    def native():
        *_, vp, q4, k4, v4, ds, packet = pack()
        v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(vp, 0.10)
        return fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, s8, route)[0][:, :, :qlen].contiguous()

    def split():
        qp, kp, vp, q4, k4, v4, ds, packet = pack()
        v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(vp, 0.10)
        tiles = vp.shape[2] // 128
        ids = torch.empty_like(route, dtype=torch.int32)
        counts = torch.empty((1, heads), device=q.device, dtype=torch.int32)
        qdec = torch.empty((1, heads, qp.shape[2], d),
                           device=q.device, dtype=torch.float16)
        kdec = torch.empty((1, heads, kp.shape[2], d),
                           device=q.device, dtype=torch.float16)
        zero_selected_v4[(tiles, heads)](v4[0], route, heads, vp.shape[2], tiles)
        compact_route[(heads,)](route, ids, counts, tiles)
        dequant_qk4[(qp.shape[2] // 32, heads)](
            q4[0], q4[1], route, qdec, qp.shape[2], tiles, False)
        dequant_qk4[(kp.shape[2] // 32, heads)](
            k4[0], k4[1], route, kdec, kp.shape[2], tiles, True)
        base_full, lse_full = fp4attn_lse_cuda.fwd(*packet)[:2]
        base = base_full[:, :, :qlen].contiguous()
        lse = lse_full[:, :, :qlen].contiguous()
        out = torch.empty_like(base)
        split_selected_fp8_pv[(triton.cdiv(qlen, args.bm), heads)](
            q4[0], k4[0], q4[1], k4[1], qdec, kdec, ds, v8, s8,
            ids, counts, lse, base, out, qlen, qp.shape[2], kp.shape[2],
            tiles, args.bm, True, num_warps=args.warps,
            num_stages=args.stages)
        return out

    with torch.inference_mode():
        prepare()
        baseline = native()
        prepare()
        candidate = split()
        vs_native = float((baseline.float() - candidate.float()).square().sum() /
                          baseline.float().square().sum())
        report = {
            "format": "e120-full-online-v1", "capture": str(CAPTURE),
            "gpu": torch.cuda.get_device_name(), "shape": list(q.shape),
            "query_block": args.bm,
            "warps": args.warps, "stages": args.stages,
            "timing_excludes_benchmark_only_k_restore": True,
            "candidate_vs_native_mixed_nmse": vs_native,
            "timing": {
                "pure_fixed_online": time_cuda(lambda:
                    sageattn3_blackwell(q, k_work, v, is_causal=False,
                                       per_block_mean=True), prepare),
                "native_sparse_mixed_online": time_cuda(native, prepare),
                "split_fp8_pv_online": time_cuda(split, prepare),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_vs_native_mixed_nmse": vs_native,
                      "timing_ms": {key: value["mean_ms"] for key, value in
                                    report["timing"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
