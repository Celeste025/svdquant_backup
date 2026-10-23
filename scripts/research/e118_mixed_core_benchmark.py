#!/usr/bin/env python3
"""Compare isolated mixed FP8-PV candidates against the corrected core."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).square().sum() / b.square().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--candidate", choices=("noinline", "inplace"),
                        default="noinline")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E118/noinline_gpu0.json")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed"),
                    str(ROOT / "scripts/research")]
    import fp4attn_cuda
    import fp4attn_mixed_maskfix_cuda
    candidate = importlib.import_module(f"fp4attn_mixed_{args.candidate}_cuda")
    import sage3_mixed_router_cuda
    import sage3_mixed_router_sparse_cuda
    from e117_native_mixed_cost_split import time_cuda
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    report = {"format": "e118-mixed-fp8-candidate-v1", "candidate": args.candidate,
              "capture": str(CAPTURE),
              "gpu": torch.cuda.get_device_name(args.gpu), "cases": {}, "timing": {}}
    with torch.inference_mode():
        for name, start, length, heads in (("control_128", 0, 128, 4),
                                           ("tail_54", 9482, 54, 56),
                                           ("main_9482", 0, 9482, 56)):
            q, k, v = (payload[key][start:start + length]
                       .to(f"cuda:{args.gpu}").transpose(0, 1)[:heads].unsqueeze(0)
                       for key in ("q", "k", "v"))
            qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
            q4 = scale_and_quant_fp4(qp)
            k4 = scale_and_quant_fp4_permute(kp)
            v4 = scale_and_quant_fp4_transpose(vp)
            packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                      ds, length, None, 128 ** -0.5, False, True, True)
            v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(vp, 0.10)
            zero = torch.zeros_like(route)
            if name != "main_9482":
                # A selected sparse tile must have initialized FP8 payload.
                route = torch.ones_like(route)
                v8, s8, _, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)

            pure = fp4attn_cuda.fwd(*packet)[0][:, :, :length]
            base_zero = fp4attn_mixed_maskfix_cuda.fwd_mixed(
                *packet, v8, s8, zero)[0][:, :, :length]
            cand_zero = candidate.fwd_mixed(
                *packet, v8, s8, zero)[0][:, :, :length]
            base_real = fp4attn_mixed_maskfix_cuda.fwd_mixed(
                *packet, v8, s8, route)[0][:, :, :length]
            cand_real = candidate.fwd_mixed(
                *packet, v8, s8, route)[0][:, :, :length]
            outputs = (pure, base_zero, cand_zero, base_real, cand_real)
            assert all(bool(torch.isfinite(out).all()) for out in outputs)
            assert torch.equal(base_zero, pure)
            row = {
                "shape": list(q.shape),
                "selected_tiles": int(route.sum()), "total_tiles": route.numel(),
                "zero_vs_pure_nmse": nmse(cand_zero, pure),
                "real_vs_baseline_nmse": nmse(cand_real, base_real),
                "real_vs_pure_nmse": nmse(cand_real, pure),
                "zero_bitwise_equal": bool(torch.equal(cand_zero, pure)),
                "real_bitwise_equal": bool(torch.equal(cand_real, base_real)),
                "zero_max_abs_diff": float((cand_zero.float() - pure.float()).abs().max()),
                "real_max_abs_diff": float((cand_real.float() - base_real.float()).abs().max()),
                "zero_equal_fraction": float((cand_zero == pure).sum().item() / pure.numel()),
                "real_equal_fraction": float((cand_real == base_real).sum().item() / base_real.numel()),
            }
            report["cases"][name] = row
            print(json.dumps({name: row}), flush=True)
            if row["zero_vs_pure_nmse"] > 1e-9 or row["real_vs_baseline_nmse"] > 1e-9:
                raise RuntimeError(f"Numerical gate failed for {name}")
            if name == "main_9482":
                report["timing"] = {
                    "pure": time_cuda(lambda: fp4attn_cuda.fwd(*packet)),
                    "baseline_zero": time_cuda(lambda:
                        fp4attn_mixed_maskfix_cuda.fwd_mixed(*packet, v8, s8, zero)),
                    "candidate_zero": time_cuda(lambda:
                        candidate.fwd_mixed(*packet, v8, s8, zero)),
                    "baseline_real": time_cuda(lambda:
                        fp4attn_mixed_maskfix_cuda.fwd_mixed(*packet, v8, s8, route)),
                    "candidate_real": time_cuda(lambda:
                        candidate.fwd_mixed(*packet, v8, s8, route)),
                }
                print(json.dumps({"timing_ms": {key: row["mean_ms"]
                                  for key, row in report["timing"].items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
