#!/usr/bin/env python3
"""Record a cross-build Sage3 output comparison, including 64-token slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def nmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref, cand = reference.float(), candidate.float()
    return float((ref - cand).square().sum() / ref.square().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref = torch.load(args.reference, map_location="cpu", weights_only=True)
    cand = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if ref["tokens"] != cand["tokens"] or ref["output"].shape != cand["output"].shape:
        raise RuntimeError("output shape or token count differs")
    a, b = ref["output"], cand["output"]
    report = {"format": "e130-sage3-cross-build-output-v1",
              "tokens": ref["tokens"], "reference_source": ref["source"],
              "candidate_source": cand["source"],
              "reference_file": str(args.reference), "candidate_file": str(args.candidate),
              "all_nmse": nmse(a, b),
              "per_64_token_nmse": [nmse(a[:, :, start:start + 64], b[:, :, start:start + 64])
                                      for start in range(0, ref["tokens"], 64)],
              "all_finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tokens": report["tokens"], "all_nmse": report["all_nmse"],
                      "first_two_slices": report["per_64_token_nmse"][:2]}, indent=2))


if __name__ == "__main__":
    main()
