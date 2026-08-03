#!/usr/bin/env python3
"""Compute and plot BF16 versus W4A4 output/latent errors (MSE + NMSE)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from quant_error_metrics import mse_nmse


ROOT = Path("outputs/rollout_latent_error")


def metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    out = mse_nmse(reference, estimate)
    x = reference.float().reshape(-1)
    y = estimate.float().reshape(-1)
    out["cosine"] = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
    out["magnitude_ratio"] = float(y.norm() / x.norm().clamp_min(1e-20))
    return out


def analyze(model: str) -> dict:
    data = {
        mode: torch.load(ROOT / f"{model}_{mode}.pt", map_location="cpu", weights_only=False)
        for mode in ("bf16", "quant", "forced")
    }
    indices = data["bf16"]["capture_indices"]
    rows = []
    for i in indices:
        rows.append(
            {
                "step": i + 1,
                "teacher_forced_output": metrics(
                    data["bf16"]["model_outputs"][i],
                    data["forced"]["model_outputs"][i],
                ),
                "rollout_output": metrics(
                    data["bf16"]["model_outputs"][i],
                    data["quant"]["model_outputs"][i],
                ),
                "post_step_latent": metrics(
                    data["bf16"]["post_latents"][i],
                    data["quant"]["post_latents"][i],
                ),
            }
        )
    return {"model": model, "rows": rows}


def main() -> None:
    reports = {model: analyze(model) for model in ("flux", "wan")}
    (ROOT / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    series = (
        ("teacher_forced_output", "Single-step output\n(same BF16 latent)"),
        ("rollout_output", "Output\n(independent rollout)"),
        ("post_step_latent", "Post-step latent\n(accumulated rollout)"),
    )
    colors = {"flux": "#277DA1", "wan": "#F94144"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    for col, (key, title) in enumerate(series):
        for model in ("flux", "wan"):
            rows = reports[model]["rows"]
            steps = [r["step"] for r in rows]
            axes[0, col].plot(
                steps,
                [r[key]["mse"] for r in rows],
                marker="o",
                ms=4,
                lw=2,
                color=colors[model],
                label="FLUX.1-dev" if model == "flux" else "Wan2.1-1.3B",
            )
            axes[1, col].plot(
                steps,
                [100 * r[key]["nmse"] for r in rows],
                marker="o",
                ms=4,
                lw=2,
                color=colors[model],
                label="FLUX.1-dev" if model == "flux" else "Wan2.1-1.3B",
            )
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("MSE")
        axes[1, col].set_ylabel("NMSE (%)")
        axes[1, col].set_xlabel("Denoising step")
        for row in range(2):
            axes[row, col].set_yscale("log")
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Exp9: BF16 vs W4A4 — MSE (top) and NMSE (bottom); group=64, rank=32, seed=44"
    )
    fig.savefig(ROOT / "wan_flux_output_latent_error.png", dpi=220)
    fig.savefig(ROOT / "wan_flux_output_latent_error_mse_nmse.png", dpi=220)
    plt.close(fig)

    summary = {}
    for model, report in reports.items():
        summary[model] = {}
        for key, _ in series:
            mse_vals = np.array([r[key]["mse"] for r in report["rows"]])
            nmse_vals = np.array([r[key]["nmse"] for r in report["rows"]])
            summary[model][key] = {
                "median_mse": float(np.median(mse_vals)),
                "max_mse": float(mse_vals.max()),
                "final_mse": float(mse_vals[-1]),
                "median_nmse": float(np.median(nmse_vals)),
                "max_nmse": float(nmse_vals.max()),
                "final_nmse": float(nmse_vals[-1]),
            }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
