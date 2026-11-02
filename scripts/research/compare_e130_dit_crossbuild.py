#!/usr/bin/env python3
"""Compare saved H3 DiT video/audio outputs from two Sage3 binary builds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    numerator = (a.double() - b.double()).square().sum()
    denominator = b.double().square().sum().clamp_min(1e-30)
    return float(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--arm", default="sage3_fixed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(directory: Path, arm: str) -> dict:
        path = directory / f"step{args.step:02d}_{arm}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["step"] != args.step or payload["arm"] != arm:
            raise RuntimeError(f"saved output identity mismatch: {path}")
        return payload

    reference = read(args.baseline_dir, "bf16_sdpa")
    candidate_reference = read(args.candidate_dir, "bf16_sdpa")
    baseline = read(args.baseline_dir, args.arm)
    candidate = read(args.candidate_dir, args.arm)
    report = {"format": "e130-h3-dit-crossbuild-v1", "step": args.step,
              "arm": args.arm, "baseline_dir": str(args.baseline_dir),
              "candidate_dir": str(args.candidate_dir), "modalities": {}}
    for name in ("video", "audio"):
        a, b = baseline[name], candidate[name]
        teacher_a, teacher_b = reference[name], candidate_reference[name]
        if a.shape != b.shape or a.shape != teacher_a.shape or a.shape != teacher_b.shape:
            raise RuntimeError(f"shape mismatch in {name}")
        if not all(bool(torch.isfinite(x).all()) for x in (a, b, teacher_a, teacher_b)):
            raise RuntimeError(f"nonfinite {name} output")
        report["modalities"][name] = {
            "shape": list(a.shape),
            "bf16_crossbuild_bitwise_equal": bool(torch.equal(teacher_a, teacher_b)),
            "candidate_vs_baseline_nmse": nmse(b, a),
            "baseline_vs_bf16_nmse": nmse(a, teacher_a),
            "candidate_vs_bf16_nmse": nmse(b, teacher_a),
            "candidate_vs_baseline_max_abs": float((b - a).abs().max()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["modalities"], indent=2))


if __name__ == "__main__":
    main()
