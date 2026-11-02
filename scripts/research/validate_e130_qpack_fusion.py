#!/usr/bin/env python3
"""Validate experimental Sage3 per-group Q-center plus FP4-pack fusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys

import torch

CAPTURE = Path(
    "/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/"
    "race/block25_qkv_h56.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--fast-raw-k-delta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    if "fast_qpack" not in inspect.signature(sageattn3_blackwell).parameters:
        raise RuntimeError("candidate fast_qpack parameter is missing")
    if not hasattr(api.fp4quant_cuda, "scaled_fp4_quant_group_center"):
        raise RuntimeError("candidate fused-Q extension is missing")
    if args.fast_raw_k_delta:
        if "fast_raw_k_delta" not in inspect.signature(sageattn3_blackwell).parameters:
            raise RuntimeError("candidate raw-K API parameter is missing")
        if not hasattr(api.fp4quant_cuda, "scaled_fp4_quant_permute_center_readonly"):
            raise RuntimeError("candidate read-only K pack extension is missing")

    torch.cuda.set_device(0)
    source = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    if source["q"].ndim == 3:
        if args.batch_size != 1 or args.heads != 56:
            raise ValueError("rank-3 capture supports only B1H56")
        max_n = source["q"].shape[0]
        make_hnd = lambda name, n: source[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
    elif source["q"].ndim == 4:
        if args.batch_size < 1 or not 1 <= args.heads <= source["q"].shape[1]:
            raise ValueError("batch/head request outside capture")
        max_n = source["q"].shape[2]
        make_hnd = lambda name, n: source[name][:, :args.heads, :n].to("cuda:0").expand(
            args.batch_size, -1, -1, -1).contiguous()
    else:
        raise RuntimeError("unsupported H3 capture")
    if any(n < 1 or n > max_n for n in args.lengths):
        raise ValueError("length exceeds capture")

    cases = []
    with torch.inference_mode():
        for n in args.lengths:
            q, k, v = [make_hnd(name, n) for name in ("q", "k", "v")]
            b, h, _, d = q.shape
            p = ((n + 127) // 128) * 128
            qp = torch.empty((b, h, p, d), device=q.device, dtype=q.dtype)
            qm = torch.empty((b, h, p // 128, d), device=q.device, dtype=q.dtype)
            qm_only = torch.empty_like(qm)
            api._e130_group_mean_pad_q[(b, h, p // 128)](
                q, qp, qm, n, d, *q.stride(), *qp.stride(), *qm.stride(), num_warps=4
            )
            api._e130_group_mean_q_only[(b, h, p // 128)](
                q, qm_only, n, d, *q.stride(), *qm_only.stride(), num_warps=4
            )
            old_code, old_scale = api.scale_and_quant_fp4(qp)
            code = torch.empty_like(old_code)
            scale = torch.empty_like(old_scale)
            api.fp4quant_cuda.scaled_fp4_quant_group_center(q, qm_only, code, scale)
            old = sageattn3_blackwell(q, k.clone(), v, fast_preprocess=True)
            new = sageattn3_blackwell(
                q, k.clone(), v, fast_preprocess=True, fast_qpack=True
            )
            combined = sageattn3_blackwell(
                q, k.clone(), v, fast_preprocess=True,
                fast_qpack=True, fast_delta_gemm=True
            )
            k_before = k.clone()
            raw_k_output = (sageattn3_blackwell(
                q, k, v, fast_preprocess=True, fast_qpack=True,
                fast_delta_gemm=True, fast_raw_k_delta=True)
                if args.fast_raw_k_delta else combined)
            row = {
                "tokens": n,
                "mean_equal": bool(torch.equal(qm, qm_only)),
                "qcode_equal": bool(torch.equal(old_code, code)),
                "qscale_equal": bool(torch.equal(old_scale, scale)),
                "output_equal": bool(torch.equal(old, new)),
                "combined_output_equal": bool(torch.equal(old, combined)),
                "output_finite": bool(torch.isfinite(new).all()),
                "combined_output_finite": bool(torch.isfinite(combined).all()),
                "raw_k_output_equal": bool(torch.equal(combined, raw_k_output)),
                "raw_k_output_finite": bool(torch.isfinite(raw_k_output).all()),
                "raw_k_keeps_k": bool(torch.equal(k, k_before)),
                "output_nmse": float((old.float() - new.float()).square().sum()
                                     / old.float().square().sum()),
            }
            row["passed"] = all(row[key] for key in (
                "mean_equal", "qcode_equal", "qscale_equal", "output_equal",
                "combined_output_equal", "output_finite", "combined_output_finite",
                "raw_k_output_equal", "raw_k_output_finite", "raw_k_keeps_k",
            ))
            print(json.dumps(row), flush=True)
            cases.append(row)
            if not row["passed"]:
                break
    report = {
        "format": "e130-qpack-fusion-validation-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.sage3_dir.resolve()),
        "capture": str(args.capture.resolve()),
        "fast_raw_k_delta": args.fast_raw_k_delta,
        "batch_size": args.batch_size, "heads": args.heads,
        "gpu": torch.cuda.get_device_name(0),
        "cases": cases,
        "passed": len(cases) == len(args.lengths) and all(row["passed"] for row in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("fused-Q validation failed")


if __name__ == "__main__":
    main()
