#!/usr/bin/env python3
"""Plot FLUX.1-dev versus Wan2.1 post-scaling activation statistics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/activation_error_comparison")
METRICS = ("p999_over_rms", "max_over_rms", "mse", "nmse")
LABELS = {
    "p999_over_rms": "P99.9(|x_scaled|) / RMS",
    "max_over_rms": "max(|x_scaled|) / RMS",
    "mse": "INT4 activation MSE",
    "nmse": "INT4 activation NMSE",
}


def aggregate_flux(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["name"]].append(record)
    result = []
    for name, values in grouped.items():
        row = {"name": name}
        for metric in values[0]:
            if isinstance(values[0][metric], (int, float)):
                row[metric] = float(np.median([v[metric] for v in values]))
        result.append(row)
    return result


def summarize(records: list[dict]) -> dict:
    result = {"num_layers": len(records)}
    for metric in (
        "rms",
        "absmax",
        "p99_over_rms",
        "p999_over_rms",
        "max_over_rms",
        "tail_gt_6rms",
        "mse",
        "nmse",
        "mae",
        "nmae",
    ):
        values = np.asarray([r[metric] for r in records])
        result[metric] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p10": float(np.quantile(values, 0.1)),
            "p90": float(np.quantile(values, 0.9)),
        }
    return result


def main() -> None:
    wan = json.loads((ROOT / "wan_metrics.json").read_text())
    flux_calls = json.loads((ROOT / "flux_metrics.json").read_text())
    flux = aggregate_flux(flux_calls)
    datasets = [flux, wan]
    names = ["FLUX.1-dev", "Wan2.1-1.3B"]
    colors = ["#3478bf", "#e56b4a"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, metric in zip(axes.flat, METRICS, strict=True):
        values = [[r[metric] for r in data] for data in datasets]
        parts = ax.violinplot(values, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors, strict=True):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.65)
        parts["cmedians"].set_color("#222222")
        ax.set_xticks([1, 2], names)
        ax.set_ylabel(LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
        if metric in ("max_over_rms", "mse"):
            ax.set_yscale("log")
        medians = [np.median(v) for v in values]
        for index, median in enumerate(medians, 1):
            ax.text(index, median, f"  {median:.3g}", va="bottom", fontsize=9)
    axes[0, 0].set_title("Residual activation outliers after SVDQuant scaling")
    axes[0, 1].set_title("Worst sampled outlier after scaling")
    axes[1, 0].set_title("Absolute dynamic-group64 INT4 error")
    axes[1, 1].set_title("Scale-normalized dynamic-group64 INT4 error")
    fig.suptitle(
        "Post-scaling activation quantization: FLUX.1-dev vs Wan2.1-1.3B",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Each violin is across projection layers; FLUX values aggregate all 50 denoising steps per probed layer.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(ROOT / "scaled_activation_error_comparison.png", dpi=180)

    summary = {
        "methodology": {
            "activation": "x_scaled = (x + optional_shift) / smooth",
            "quantizer": "dynamic INT4 group size 64; signed except Wan shifted FFN-down groups use unsigned",
            "flux_sampling": "4 dual-stream blocks (0,6,12,18), 8 projection families, all 50 steps, 64 tokens/call",
            "wan_sampling": "all 210 calibrated projection groups, 2048 stratified tokens/group from 8 prompts x 10 steps",
            "note": "MSE is scale-dependent; NMSE and outlier/RMS ratios are the primary cross-model comparison.",
        },
        "FLUX.1-dev": summarize(flux),
        "Wan2.1-1.3B": summarize(wan),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
