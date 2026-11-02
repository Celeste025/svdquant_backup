#!/usr/bin/env python3
"""Validate E137 axis permutations on saved paired DiT output tensors (CPU only)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


GRID = (37, 18, 32)
CASES = {"race": (Path("results/research/E103/manifest.json"), 20026),
         "clap0": (Path("results/research/E104/manifest.json"), 38001610)}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.sum(left * right, dtype=np.float64))
    denominator = math.sqrt(float(np.sum(left * left, dtype=np.float64)) *
                            float(np.sum(right * right, dtype=np.float64)))
    return numerator / denominator if denominator else 0.0


def adjacency(error: np.ndarray) -> dict[str, float]:
    time, height, width = GRID
    shaped = error.reshape(time, height, width, 96)
    centered = shaped - shaped.mean(axis=(1, 2), keepdims=True)
    return {
        "spatial_h_signed_vector_cosine": cosine(centered[:, 1:], centered[:, :-1]),
        "spatial_w_signed_vector_cosine": cosine(centered[:, :, 1:], centered[:, :, :-1]),
        "temporal_signed_vector_cosine": cosine(centered[1:], centered[:-1]),
    }


def permute(error: np.ndarray, arm: str, seed: int, step: int) -> np.ndarray:
    time, height, width = GRID
    shaped = error.reshape(time, height, width, 96)
    generator = np.random.default_rng(seed + (137_000_000 if arm == "spatial_shared" else 138_000_000) + step)
    if arm == "spatial_shared":
        indices = generator.permutation(height * width)
        return shaped.reshape(time, height * width, 96)[:, indices].reshape(error.shape)
    if arm == "temporal_frame":
        return shaped[generator.permutation(time)].reshape(error.shape)
    raise ValueError(arm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/research/E137/axis_intervention_contract.json"))
    args = parser.parse_args()
    cases: dict[str, object] = {}
    for name, (manifest_path, seed) in CASES.items():
        manifest = json.loads(manifest_path.read_text())
        rows = []
        for record in manifest["records"]:
            step = int(record["step"])
            payload = torch.load(record["path"], map_location="cpu", weights_only=False)
            error = payload["video_error_fp32"].numpy()
            if error.shape != (np.prod(GRID), 96):
                raise RuntimeError(f"{name}/step {step}: unexpected error shape {error.shape}")
            original = adjacency(error)
            transforms = {}
            for arm in ("spatial_shared", "temporal_frame"):
                changed = permute(error, arm, seed, step)
                if not np.array_equal(np.sort(changed.reshape(-1, 96), axis=0),
                                      np.sort(error.reshape(-1, 96), axis=0)):
                    raise RuntimeError(f"{name}/step {step}/{arm}: token vector multiset changed")
                transforms[arm] = adjacency(changed)
            rows.append({"step": step, "original": original, "transforms": transforms})
        cases[name] = {"prompt_id": manifest["prompt_id"], "seed": seed, "steps": rows}
    report = {
        "format": "h3-e137-axis-intervention-contract-v1",
        "purpose": "CPU validation only on saved BF16-trajectory outputs; not a closed-loop quality result",
        "video_grid_thw": list(GRID),
        "invariants": {
            "spatial_shared": "preserves each token's complete 96D error vector and its temporal path; changes spatial coordinate",
            "temporal_frame": "preserves each frame's complete spatial error field; changes temporal coordinate",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
