#!/usr/bin/env python3
"""Summarize valid E137 error-placement rollouts without treating RGB NMSE as quality."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np


ROOT = Path("/home/wjq/workspace/svdquant-exp")
E105 = ROOT / "results/research/E105"
DATA = Path("/data1/models/svdquant-wjq/research/20261030/E137_h3_error_axis_permutation")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rgb_nmse(reference: Path, candidate: Path) -> dict:
    ref_c, out_c = av.open(str(reference)), av.open(str(candidate))
    numer = denom = 0.0
    n = 0
    try:
        for rf, of in zip(ref_c.decode(video=0), out_c.decode(video=0)):
            r = rf.to_ndarray(format="rgb24").astype(np.float64)
            o = of.to_ndarray(format="rgb24").astype(np.float64)
            numer += float(np.square(o - r).sum())
            denom += float(np.square(r).sum())
            n += 1
        # zip silently truncates, so independently check frame counts.
        ref_total = sum(1 for _ in av.open(str(reference)).decode(video=0))
        out_total = sum(1 for _ in av.open(str(candidate)).decode(video=0))
    finally:
        ref_c.close()
        out_c.close()
    if not n or n != ref_total or n != out_total:
        raise RuntimeError(f"frame mismatch: shared={n}, ref={ref_total}, candidate={out_total}")
    return {"frames": n, "rgb_nmse": numer / denom, "rgb_mae": None}


def max_gate_error(report: dict) -> dict:
    vals = [
        abs(float(mod["relative_mse_change"]))
        for step in report["steps"] for mod in step["modalities"].values()
    ]
    return {"max_abs_relative_mse_change": max(vals), "gate_limit": 0.005,
            "passes": max(vals) <= 0.005}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--case", choices=("race", "clap0"), required=True)
    args = ap.parse_args()
    baseline = json.loads((E105 / args.case / "five_arm_summary.json").read_text())
    root = ROOT / "results/research/E137" / args.attempt / args.case
    rows = []
    for arm in ("spatial_shared", "temporal_frame"):
        denoise_path = root / f"{arm}_denoise.json"
        video_path = root / f"{arm}_video.json"
        if not (denoise_path.exists() and video_path.exists()):
            continue
        denoise, video = json.loads(denoise_path.read_text()), json.loads(video_path.read_text())
        # Decode reports are deliberately small provenance records and use the
        # presence of the verified media path rather than a separate status key.
        if denoise.get("status") != "denoise_complete" or "video" not in video:
            continue
        generated = Path(video["video"])
        metric = rgb_nmse(Path(baseline["sources"]["bf16"]), generated)
        original = float(baseline["rollout_distance_proxy"]["svdquant"]["rgb_nmse"])
        rows.append({
            "arm": arm,
            "video_path": str(generated),
            "video_sha256": sha256(generated),
            "error_energy_gate": max_gate_error(denoise),
            "decoded_rgb_distance_to_bf16": metric,
            "relative_reduction_vs_original_svdquant": 1.0 - metric["rgb_nmse"] / original,
        })
    output = {
        "format": "h3-e137-axis-permutation-summary-v1",
        "case": args.case,
        "definition": "decoded H.264 RGB distance to paired BF16 free rollout; a divergence proxy, not perceptual quality",
        "baseline": {"bf16": baseline["sources"]["bf16"], "svdquant": baseline["sources"]["svdquant"],
                     "original_svdquant_rgb_nmse": baseline["rollout_distance_proxy"]["svdquant"]["rgb_nmse"]},
        "complete_arms": rows,
        "incomplete_arms_are_excluded": True,
    }
    out = root / "axis_permutation_summary.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
