#!/usr/bin/env python3
"""Compare the signed H3 paired-output error with a unit Gaussian.

Each video-output element is centered and scaled by its own denoising step's
mean and standard deviation before pooling steps. Counts use all elements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def accumulate(manifest_path: Path, bins: np.ndarray, thresholds: np.ndarray) -> dict:
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    if len(records) != 20 or [r["step"] for r in records] != list(range(20)):
        raise RuntimeError(f"not a complete 20-step capture: {manifest_path}")
    histogram = np.zeros(len(bins) - 1, dtype=np.int64)
    exceedances = np.zeros(len(thresholds), dtype=np.int64)
    n = zero_count = 0
    per_step = []
    for record in records:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"raw file failed integrity check: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["step"] != record["step"]:
            raise RuntimeError(f"step metadata mismatch: {path}")
        values = payload["video_error_fp32"].numpy().ravel().astype(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite errors: {path}")
        mean, std = float(values.mean()), float(values.std())
        if std <= 0:
            raise RuntimeError(f"zero error variance: {path}")
        z = (values - mean) / std
        az = np.sort(np.abs(z))
        histogram += np.histogram(z, bins=bins)[0]
        exceedances += len(z) - np.searchsorted(az, thresholds, side="right")
        n += len(z)
        zero_count += int(np.count_nonzero(values == 0))
        per_step.append({"step": record["step"], "mean": mean, "std": std,
                         "count": len(z), "gt3_fraction": float((az > 3).mean()),
                         "gt4_fraction": float((az > 4).mean())})
    return {"prompt_id": manifest["prompt_id"], "seed": manifest["seed"],
            "source_manifest": str(manifest_path.resolve()), "count": n,
            "exact_zero_fraction": zero_count / n, "histogram": histogram,
            "exceedances": exceedances, "per_step": per_step}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-manifest", type=Path, required=True)
    parser.add_argument("--clap-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    bins = np.linspace(-8, 8, 321)
    centers = (bins[:-1] + bins[1:]) / 2
    thresholds = np.linspace(0, 8, 161)
    cases = {
        "race": accumulate(args.race_manifest, bins, thresholds),
        "clap": accumulate(args.clap_manifest, bins, thresholds),
    }
    gaussian_pdf = np.exp(-centers**2 / 2) / math.sqrt(2 * math.pi)
    gaussian_survival = np.array([math.erfc(float(t) / math.sqrt(2)) for t in thresholds])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.80, wspace=0.20)
    palette = {"race": "#22577a", "clap": "#c25b2a"}
    labels = {"race": "Race · prompt 16", "clap": "Clap · prompt 161"}
    for key, case in cases.items():
        density = case["histogram"] / (case["count"] * np.diff(bins))
        axes[0].plot(centers, density, color=palette[key], lw=1.8, label=labels[key])
        survival = case["exceedances"] / case["count"]
        axes[1].plot(thresholds, survival, color=palette[key], lw=1.9,
                     label=f"{labels[key]} · P(|z|>4)={100*survival[80]:.3f}%")
    axes[0].plot(centers, gaussian_pdf, color="#454545", ls="--", lw=1.6,
                 label="Standard normal")
    axes[1].plot(thresholds, gaussian_survival, color="#454545", ls="--", lw=1.6,
                 label=f"Standard normal · P(|z|>4)={100*gaussian_survival[80]:.4f}%")
    axes[0].set(xlim=(-5, 5), ylim=(0, None), xlabel="Signed error z-score", ylabel="Probability density",
                title="Central shape · signed errors")
    axes[1].set(xlim=(0, 8), ylim=(1e-5, 1), xlabel="Threshold |z|", ylabel="P(|z| > threshold)",
                yscale="log", title="Two-sided tail · logarithmic probability")
    axes[1].axvline(4, color="#777777", ls=":", lw=0.9)
    for ax in axes:
        ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.8)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("H3 video DiT output error vs Gaussian · each denoising step standardized separately",
                 fontsize=13, y=0.965)
    fig.text(0.5, 0.065,
             "All 20 paired BF16-trajectory steps per case; each step has 2,045,952 output elements. "
             "Zero-valued BF16 differences are included.",
             ha="center", va="bottom", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    serializable = {"definition": "z=(SVDQuant-BF16-step_mean)/step_std, video outputs only",
                    "bin_edges": bins.tolist(), "tail_thresholds": thresholds.tolist(),
                    "standard_normal_gt3_fraction": float(gaussian_survival[60]),
                    "standard_normal_gt4_fraction": float(gaussian_survival[80]),
                    "cases": {key: {**{k: v for k, v in value.items()
                                        if k not in ("histogram", "exceedances")},
                                    "histogram_counts": value["histogram"].tolist(),
                                    "tail_exceedance_counts": value["exceedances"].tolist()}
                              for key, value in cases.items()}}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(serializable, indent=2, allow_nan=False) + "\n")
    print(args.output)
    for key, case in cases.items():
        print(key, "elements", case["count"], "P(|z|>3)", case["exceedances"][60]/case["count"],
              "P(|z|>4)", case["exceedances"][80]/case["count"])


if __name__ == "__main__":
    main()
