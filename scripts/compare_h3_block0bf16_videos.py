#!/usr/bin/env python3
"""Frame-level comparison: block0-restored SVDQuant vs BF16 and SVDQuant baselines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np


def frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                arr = frame.to_ndarray(format="rgb24")
                yield arr


def frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in frames(path))


def paired_metrics(a_path: Path, b_path: Path) -> dict:
    psnrs, maes = [], []
    for fa, fb in zip(frames(a_path), frames(b_path), strict=False):
        if fa.shape != fb.shape:
            raise RuntimeError(f"shape mismatch: {fa.shape} vs {fb.shape}")
        d = fa.astype(np.float32) - fb.astype(np.float32)
        mse = float((d * d).mean())
        psnrs.append(10.0 * np.log10(255.0 * 255.0 / max(mse, 1e-12)))
        maes.append(float(np.abs(d).mean()))
    if not psnrs:
        raise RuntimeError(f"no frames decoded from {a_path} / {b_path}")
    return {"frames_compared": len(psnrs), "psnr_mean": float(np.mean(psnrs)),
            "psnr_min": float(np.min(psnrs)), "psnr_max": float(np.max(psnrs)),
            "mae_mean": float(np.mean(maes))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--svdquant", type=Path, required=True)
    p.add_argument("--block0bf16", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    sq_vs_bf16 = paired_metrics(args.svdquant, args.bf16)
    b0_vs_bf16 = paired_metrics(args.block0bf16, args.bf16)
    b0_vs_sq = paired_metrics(args.block0bf16, args.svdquant)

    wins = total = 0
    for fa, f_sq, f_b0 in zip(frames(args.bf16), frames(args.svdquant),
                              frames(args.block0bf16), strict=False):
        d_sq = fa.astype(np.float32) - f_sq.astype(np.float32)
        d_b0 = fa.astype(np.float32) - f_b0.astype(np.float32)
        wins += float((d_b0 * d_b0).mean()) < float((d_sq * d_sq).mean())
        total += 1
    report = {"case": args.case, "bf16": str(args.bf16), "svdquant": str(args.svdquant),
              "block0bf16": str(args.block0bf16),
              "frame_counts": {"bf16": frame_count(args.bf16), "svdquant": frame_count(args.svdquant),
                               "block0bf16": frame_count(args.block0bf16)},
              "svdquant_vs_bf16": sq_vs_bf16, "block0bf16_vs_bf16": b0_vs_bf16,
              "block0bf16_vs_svdquant": b0_vs_sq,
              "block0bf16_closer_to_bf16_frames": f"{wins}/{total}"}
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
