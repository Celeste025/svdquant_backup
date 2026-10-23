#!/usr/bin/env python3
"""Validate exported FP4 softmax LSE and benchmark its incremental cost."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def main() -> None:
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "scripts/research")]
    import fp4attn_cuda
    import fp4attn_lse_cuda
    from e117_native_mixed_cost_split import time_cuda
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    report = {"format": "e119-lse-validation-v1", "capture": str(CAPTURE),
              "gpu": torch.cuda.get_device_name(), "cases": {}, "timing": {}}
    with torch.inference_mode():
        for name, start, length, heads in (("control_128", 0, 128, 4),
                                           ("tail_54", 9482, 54, 4),
                                           ("main_9482", 0, 9482, 56)):
            q, k, v = (payload[key][start:start + length]
                       .to("cuda:0").transpose(0, 1)[:heads].unsqueeze(0)
                       for key in ("q", "k", "v"))
            qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
            q4 = scale_and_quant_fp4(qp)
            k4 = scale_and_quant_fp4_permute(kp)
            v4 = scale_and_quant_fp4_transpose(vp)
            packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                      ds, length, None, 128 ** -0.5, False, True, True)
            base_o = fp4attn_cuda.fwd(*packet)[0][:, :, :length]
            new_o, lse = fp4attn_lse_cuda.fwd(*packet)[:2]
            new_o = new_o[:, :, :length]
            lse = lse[:, :, :length]
            if name == "main_9482":
                sampled = torch.tensor([0, 1, 13, 127, 128, 500, 1024, 3000,
                                        4096, 7000, 9000, 9481], device="cuda:0")
                q_sample = q[:, :, sampled].float()
                lse_sample = lse[:, :, sampled]
            else:
                q_sample = q.float()
                lse_sample = lse
            scores = torch.matmul(q_sample, kp[:, :, :length].float().transpose(-1, -2))
            scores *= 128 ** -0.5
            reference_lse = torch.logsumexp(scores, dim=-1)
            diff = (lse_sample - reference_lse).abs()
            row = {
                "shape": list(q.shape), "output_bitwise_equal": bool(torch.equal(base_o, new_o)),
                "lse_finite": bool(torch.isfinite(lse).all()),
                "lse_reference_mae": float(diff.mean()),
                "lse_reference_p99_abs": float(torch.quantile(diff.flatten(), 0.99)),
                "lse_reference_max_abs": float(diff.max()),
                "lse_range": [float(lse.min()), float(lse.max())],
                "reference_range": [float(reference_lse.min()), float(reference_lse.max())],
            }
            report["cases"][name] = row
            print(json.dumps({name: row}), flush=True)
            if not row["output_bitwise_equal"] or not row["lse_finite"]:
                raise RuntimeError(f"LSE export gate failed: {name}")
            if name == "main_9482":
                report["timing"] = {
                    "original_fp4_core": time_cuda(lambda: fp4attn_cuda.fwd(*packet)),
                    "fp4_lse_core": time_cuda(lambda: fp4attn_lse_cuda.fwd(*packet)),
                }
                print(json.dumps({"timing_ms": {key: value["mean_ms"]
                                  for key, value in report["timing"].items()}}), flush=True)
    dest = ROOT / "results/research/E119/lse_validation_gpu0.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
