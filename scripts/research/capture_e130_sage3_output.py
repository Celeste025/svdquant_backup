#!/usr/bin/env python3
"""Save one Sage3 output for cross-build numerical comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--probe", choices=("actual", "zero_qk", "ones_v", "zero_qk_ones_v"),
                        default="actual")
    parser.add_argument("--head-start", type=int, default=0)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--qk-scale", type=float, default=1.0)
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens not in (127, 128, 129, 255, 256, 257,
                           1024, 1025, 9482, 22677):
        raise ValueError("unexpected validation length")
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    total_heads = payload["q"].shape[1]
    head_count = args.heads or total_heads - args.head_start
    if not (0 <= args.head_start < total_heads and
            1 <= head_count <= total_heads - args.head_start):
        raise ValueError("head slice outside capture")
    head_slice = slice(args.head_start, args.head_start + head_count)
    if payload["q"].ndim == 3:
        q, k, v = [payload[name][:args.tokens, head_slice].to("cuda:0").transpose(0, 1).unsqueeze(0)
                   for name in ("q", "k", "v")]
    elif payload["q"].ndim == 4:
        q, k, v = [payload[name][:, head_slice, :args.tokens].to("cuda:0")
                   for name in ("q", "k", "v")]
    else:
        raise ValueError("unsupported capture rank")
    if args.probe in ("zero_qk", "zero_qk_ones_v"):
        q, k = torch.zeros_like(q), torch.zeros_like(k)
    if args.qk_scale != 1.0:
        q, k = q * args.qk_scale, k * args.qk_scale
    if args.probe in ("ones_v", "zero_qk_ones_v"):
        v = torch.ones_like(v)
    with torch.inference_mode():
        result = sageattn3_blackwell(q, k.clone(), v,
                                     is_causal=False, per_block_mean=True)
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite Sage3 output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.hash_only:
        view = result.contiguous().view(torch.uint8).cpu().numpy()
        report = {"tokens": args.tokens, "source": str(args.sage3_dir.resolve()),
                  "probe": args.probe,
                  "head_start": args.head_start, "heads": head_count,
                  "qk_scale": args.qk_scale,
                  "capture": str(args.capture.resolve()), "shape": list(result.shape),
                  "dtype": str(result.dtype), "sha256": hashlib.sha256(memoryview(view)).hexdigest()}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        torch.save({"tokens": args.tokens, "source": str(args.sage3_dir.resolve()),
                    "probe": args.probe,
                    "head_start": args.head_start, "heads": head_count,
                    "qk_scale": args.qk_scale,
                    "output": result.cpu()}, args.output)
    print(args.output, result.shape, flush=True)


if __name__ == "__main__":
    main()
