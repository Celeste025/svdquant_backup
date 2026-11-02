#!/usr/bin/env python3
"""Check E130 fused Sage3 preprocessing across full and partial K tiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research"),
                str(ROOT / "third_party/sage3_h3_maskfix")]

from bench_h3_sage_real_gpu0 import metrics
from e130_sage3_preprocess import attention

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt")
LONG_CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")
OUTPUT = ROOT / "results/research/E130/preprocess_shape_validation_gpu2.json"
LENGTHS = (127, 128, 129, 255, 256, 257, 1024, 1025, 9482)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path,
                        default=ROOT / "third_party/sage3_h3_maskfix")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--delta-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--exact-delta", action="store_true")
    parser.add_argument("--fuse-k-center-pad", action="store_true")
    parser.add_argument("--mask-padded-delta", action="store_true", default=None)
    parser.add_argument("--require-bitwise", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    results = []
    with torch.inference_mode():
        for n in LENGTHS:
            q, k, v = [payload[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
                       for name in ("q", "k", "v")]
            reference = sageattn3_blackwell(q, k.clone(), v,
                                            is_causal=False, per_block_mean=True)
            candidate = attention(q, k.clone(), v, api,
                                  exact_delta=args.exact_delta,
                                  fuse_k_center_pad=args.fuse_k_center_pad,
                                  mask_padded_delta=args.mask_padded_delta,
                                  delta_dtype=(torch.bfloat16 if args.delta_dtype == "bf16"
                                               else torch.float32))
            comparison = metrics(candidate, reference)
            bitwise = bool(torch.equal(candidate, reference))
            row = {"tokens": n, "shape": list(q.shape), "vs_official": comparison,
                   "finite": bool(torch.isfinite(candidate).all()),
                   "bitwise_equal": bitwise,
                   "passed": (bitwise if args.require_bitwise else comparison["nmse"] <= 1e-7)
                   and bool(torch.isfinite(candidate).all())}
            print(json.dumps(row), flush=True)
            results.append(row)
            del q, k, v, reference, candidate
        long_payload = torch.load(LONG_CAPTURE, map_location="cpu", weights_only=True, mmap=True)
        n = 22677
        q, k, v = [long_payload[name].to("cuda:0") for name in ("q", "k", "v")]
        if any(t.shape != (1, 56, n, 128) for t in (q, k, v)):
            raise RuntimeError("unexpected long H3 capture shape")
        reference = sageattn3_blackwell(q, k.clone(), v,
                                        is_causal=False, per_block_mean=True)
        candidate = attention(q, k.clone(), v, api,
                              exact_delta=args.exact_delta,
                              fuse_k_center_pad=args.fuse_k_center_pad,
                              mask_padded_delta=args.mask_padded_delta,
                              delta_dtype=(torch.bfloat16 if args.delta_dtype == "bf16"
                                           else torch.float32))
        comparison = metrics(candidate, reference)
        bitwise = bool(torch.equal(candidate, reference))
        row = {"tokens": n, "shape": list(q.shape), "vs_official": comparison,
               "finite": bool(torch.isfinite(candidate).all()),
               "bitwise_equal": bitwise,
               "passed": (bitwise if args.require_bitwise else comparison["nmse"] <= 1e-7)
               and bool(torch.isfinite(candidate).all())}
        print(json.dumps(row), flush=True)
        results.append(row)
    report = {"format": "e130-sage3-preprocess-shape-validation-v1",
              "gpu": torch.cuda.get_device_name(), "captures": [str(CAPTURE), str(LONG_CAPTURE)],
              "source": str(args.sage3_dir.resolve()),
              "delta_dtype": args.delta_dtype, "exact_delta": args.exact_delta,
              "fuse_k_center_pad": args.fuse_k_center_pad,
              "mask_padded_delta": args.mask_padded_delta,
              "require_bitwise": args.require_bitwise,
              "results": results, "passed": all(row["passed"] for row in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("one or more shapes failed numerical validation")


if __name__ == "__main__":
    main()
