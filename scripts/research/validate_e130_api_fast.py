#!/usr/bin/env python3
"""Validate the opt-in Sage3 preprocessing API on H3 length boundaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys

import torch

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")
LENGTHS = (127, 128, 129, 255, 256, 257, 1024, 1025, 9482, 22677)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--fast-delta-gemm", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import fp4attn_cuda
    if args.fast_delta_gemm and "fast_delta_gemm" not in inspect.signature(
            sageattn3_blackwell).parameters:
        raise RuntimeError("requested tiled delta API is absent; kwargs would silently ignore it")

    torch.cuda.set_device(0)
    source = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    if source["q"].ndim == 4:
        if args.batch_size < 1 or not 1 <= args.heads <= source["q"].shape[1]:
            raise ValueError("batch/head request exceeds capture")
        max_n = source["q"].shape[2]
        make_hnd = lambda name, n: source[name][:, :args.heads, :n, :].to("cuda:0").expand(
            args.batch_size, -1, -1, -1).contiguous()
    elif source["q"].ndim == 3:
        if args.batch_size != 1 or args.heads != 56:
            raise ValueError("rank-3 capture only supports B1H56")
        max_n = 9482
        make_hnd = lambda name, n: source[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
    else:
        raise RuntimeError("unexpected QKV capture rank")
    if any(n < 1 or n > max_n for n in args.lengths):
        raise ValueError("lengths exceed the captured QKV")
    rows = []
    with torch.inference_mode():
        for n in args.lengths:
            q, k, v = [make_hnd(name, n) for name in ("q", "k", "v")]
            k_before = k.clone()
            original = sageattn3_blackwell(q, k.clone(), v,
                                            is_causal=False, per_block_mean=True)
            fast = sageattn3_blackwell(q, k, v, is_causal=False,
                                       per_block_mean=True, fast_preprocess=True,
                                       fast_delta_gemm=args.fast_delta_gemm)
            row = {"tokens": n, "bitwise_equal": bool(torch.equal(original, fast)),
                   "fast_keeps_k": bool(torch.equal(k, k_before)),
                   "finite": bool(torch.isfinite(fast).all()),
                   "shape": list(fast.shape)}
            row["passed"] = (row["bitwise_equal"] and row["fast_keeps_k"]
                             and row["finite"] and row["shape"] == [args.batch_size, args.heads, n, 128])
            print(json.dumps(row), flush=True)
            rows.append(row)
            if not row["passed"]:
                break
    report = {"format": "e130-api-fast-validation-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(args.sage3_dir.resolve()),
              "attention_extension": str(Path(fp4attn_cuda.__file__).resolve()),
              "delta_s_dtype": getattr(fp4attn_cuda, "delta_s_dtype", "fp32"),
              "fast_delta_gemm": args.fast_delta_gemm,
              "batch_size": args.batch_size, "heads": args.heads,
              "capture": str(args.capture.resolve()), "gpu": torch.cuda.get_device_name(0),
              "cases": rows, "passed": len(rows) == len(args.lengths) and all(r["passed"] for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("fast API regression failed")


if __name__ == "__main__":
    main()
