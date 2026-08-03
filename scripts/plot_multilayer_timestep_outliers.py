#!/usr/bin/env python3
"""Plot multi-depth, multi-layer Fig.1-style timestep outlier diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("outputs/timestep_outliers_multilayer")
TYPE_ORDER = ("attention_q", "attention_out", "ffn_up", "ffn_down")
TYPE_LABEL = {
    "attention_q": "qkv_proj (to_q)",
    "attention_out": "o_proj",
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
}


def metrics(surface: torch.Tensor) -> dict:
    # surface already stores log2(absmax over tokens)
    timestep_p99 = torch.quantile(surface, 0.99, dim=1)
    timestep_median = surface.median(dim=1).values
    per_channel_range = surface.amax(dim=0) - surface.amin(dim=0)
    topk = max(1, round(surface.shape[1] * 0.01))
    sets = [set(torch.topk(row, topk).indices.tolist()) for row in surface]
    jaccard = [
        len(a & b) / max(1, len(a | b)) for a, b in zip(sets[:-1], sets[1:])
    ]
    return {
        "p99_log2_range": float(timestep_p99.max() - timestep_p99.min()),
        "median_log2_range": float(timestep_median.max() - timestep_median.min()),
        "channel_timestep_range_median": float(per_channel_range.median()),
        "channel_timestep_range_p90": float(torch.quantile(per_channel_range, 0.9)),
        "early5_late5_p99_linear_ratio": float(
            torch.pow(2.0, timestep_p99[:5].mean() - timestep_p99[-5:].mean())
        ),
        "top1pct_consecutive_jaccard": float(np.mean(jaccard)),
    }


def plot_model(model: str, payload: dict) -> dict:
    layers = payload["layers"]
    surfaces = payload["surfaces"]
    blocks = sorted({v["block"] for v in layers.values()})
    all_values = torch.cat([x.flatten() for x in surfaces.values()])
    # Floor at 0 (log2 absmax < 0 <=> absmax < 1); keep a robust high end.
    vmin = 0.0
    vmax = float(torch.quantile(all_values.clamp_min(0), 0.995))

    fig, axes = plt.subplots(
        len(TYPE_ORDER), len(blocks), figsize=(15, 11), constrained_layout=True
    )
    reports = {}
    image = None
    for row, layer_type in enumerate(TYPE_ORDER):
        for col, block in enumerate(blocks):
            key = f"{layer_type}_b{block}"
            surface = surfaces[key].clamp_min(0)
            reports[key] = {**layers[key], **metrics(surfaces[key])}
            image = axes[row, col].imshow(
                surface.T.numpy(),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                extent=(1, 50, 0, surface.shape[1]),
            )
            axes[row, col].set_title(f"{TYPE_LABEL[layer_type]} — block {block}")
            axes[row, col].set_xlabel("Denoising step")
            axes[row, col].set_ylabel("Output channel")
    title = "Wan2.1-T2V-1.3B" if model == "wan" else "FLUX.1-dev"
    fig.suptitle(
        f"{title}: per-channel timestep activation outliers\n"
        r"$\log_2(\max_{batch,tokens}|layer\ output|)$",
        fontsize=15,
    )
    fig.colorbar(image, ax=axes, shrink=0.75, label="log2(absmax)")
    fig.savefig(ROOT / f"{model}_multilayer_surfaces.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(blocks)))
    for ax, layer_type in zip(axes, TYPE_ORDER):
        for color, block in zip(colors, blocks):
            surface = surfaces[f"{layer_type}_b{block}"]
            p99 = torch.quantile(surface, 0.99, dim=1)
            ax.plot(range(1, 51), p99, color=color, label=f"block {block}", lw=2)
        ax.set_title(TYPE_LABEL[layer_type])
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("channel P99 log2(absmax)")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{title}: robust outlier envelope across timesteps")
    fig.savefig(ROOT / f"{model}_multilayer_p99_curves.png", dpi=200)
    plt.close(fig)
    return reports


def plot_a4_timestep() -> dict:
    """Plot postsmooth A4 MSE/NMSE vs timestep for matched layers."""
    colors = {"flux": "#277DA1", "wan": "#F94144"}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    summary = {}
    for model in ("flux", "wan"):
        path = ROOT / f"{model}_a4_timestep_metrics.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        summary[model] = {
            "median_mse": float(np.median([r["mse"] for r in rows])),
            "median_nmse": float(np.median([r["nmse"] for r in rows])),
        }
        for col, layer_type in enumerate(TYPE_ORDER):
            blocks = sorted({r["block"] for r in rows if r["type"] == layer_type})
            for index, block in enumerate(blocks):
                series = sorted(
                    [r for r in rows if r["type"] == layer_type and r["block"] == block],
                    key=lambda r: r["step"],
                )
                style = ("-", "--", ":")[index]
                label = f"{model.upper()} b{block}"
                axes[0, col].plot(
                    [r["step"] for r in series],
                    [r["mse"] for r in series],
                    color=colors[model],
                    ls=style,
                    marker="o",
                    ms=2.5,
                    label=label,
                )
                axes[1, col].plot(
                    [r["step"] for r in series],
                    [100 * r["nmse"] for r in series],
                    color=colors[model],
                    ls=style,
                    marker="o",
                    ms=2.5,
                    label=label,
                )
            axes[0, col].set_title(TYPE_LABEL[layer_type])
            axes[0, col].set_ylabel("A4 MSE")
            axes[0, col].set_yscale("log")
            axes[1, col].set_ylabel("A4 NMSE (%)")
            axes[1, col].set_yscale("log")
            axes[1, col].set_xlabel("Denoising step")
            for row in range(2):
                axes[row, col].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Exp8 multilayer postsmooth A4 — MSE (top) / NMSE (bottom)")
    fig.savefig(ROOT / "multilayer_a4_mse_nmse.png", dpi=200)
    plt.close(fig)
    (ROOT / "a4_timestep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    reports = {}
    for model in ("wan", "flux"):
        payload = torch.load(
            ROOT / f"{model}_multilayer_surfaces.pt",
            map_location="cpu",
            weights_only=False,
        )
        reports[model] = plot_model(model, payload)
    (ROOT / "multilayer_metrics.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    a4_summary = plot_a4_timestep()
    print(json.dumps({"outlier_metrics": reports, "a4_summary": a4_summary}, indent=2))


if __name__ == "__main__":
    main()
