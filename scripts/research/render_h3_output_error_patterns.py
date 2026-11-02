#!/usr/bin/env python3
"""Render reviewed E103 numeric reductions as standalone research figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE, ORANGE, GOLD = "#22577a", "#c25b2a", "#aa8300"


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--arrays", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text())
    if summary["format"] != "h3-e103-output-error-pattern-summary-v1" or len(summary["steps"]) != 20:
        raise RuntimeError("incomplete analysis summary")
    arrays = np.load(args.arrays)
    steps = np.arange(20)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, layout="constrained")
    axes[0].plot(steps, [100 * s["video_nmse"] for s in summary["steps"]],
                 color=BLUE, marker="o", lw=2, ms=3)
    axes[0].set_ylabel("Video output NMSE (%)")
    axes[0].set_title(f"H3 prompt {json.loads(Path(summary['manifest']).read_text())['prompt_id']}: paired DiT output error across denoising steps")
    axes[1].plot(steps, [100 * s["top_1pct_token_error_energy_share"] for s in summary["steps"]],
                 color=ORANGE, marker="o", lw=2, ms=3, label="Error energy in top 1% tokens")
    axes[1].plot(steps, [100 * s["signal_energy_share_at_same_top_tokens"] for s in summary["steps"]],
                 color=BLUE, ls="--", lw=1.8, label="BF16 energy in those tokens")
    axes[1].set_ylabel("Energy share (%)")
    axes[1].set_xlabel("Denoising step (0–19)")
    axes[1].set_xticks(range(0, 20, 2))
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, args.output_dir / "step_nmse_and_token_concentration.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for axis, (kind, keys, ylabel) in enumerate((
        ("magnitude", ("spatial_h_magnitude_corr_frame_demeaned",
                       "spatial_w_magnitude_corr_frame_demeaned",
                       "temporal_magnitude_corr_frame_demeaned"),
         "Log error-RMS neighbor correlation"),
        ("signed", ("signed_error_neighbor_cosine_h", "signed_error_neighbor_cosine_w",
                    "signed_error_neighbor_cosine_t"),
         "Signed error-vector neighbor cosine"),
    )):
        for label, key, color in zip(("height", "width", "time"), keys,
                                     (BLUE, ORANGE, GOLD), strict=True):
            axes[axis].plot(steps, [s[key] for s in summary["steps"]],
                            label=label, color=color, lw=1.8, marker="o", ms=2.5)
            if axis == 0:
                reference_key = "reference_" + key
                axes[axis].plot(steps, [s[reference_key] for s in summary["steps"]],
                                color=color, lw=1.0, ls="--", alpha=0.65)
        axes[axis].axhline(0, color="#555555", lw=0.9, ls="--")
        axes[axis].set_ylabel(ylabel)
        axes[axis].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].set_title("Adjacent latent-token structure (solid: error; dashed: BF16 signal)")
    axes[-1].set_xlabel("Denoising step (0–19)")
    axes[-1].set_xticks(range(0, 20, 2))
    save(fig, args.output_dir / "neighbor_error_structure.png")

    ce = arrays["channel_error_energy"]
    cs = arrays["channel_signal_energy"]
    nmse_channels = 100 * ce / np.maximum(cs, 1e-30)
    pooled_e = ce.sum(axis=0)
    pooled_s = cs.sum(axis=0)
    ratio = (pooled_e / pooled_e.sum()) / (pooled_s / pooled_s.sum())
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), layout="constrained",
                             gridspec_kw={"height_ratios": [2, 1]})
    vmax = float(np.quantile(nmse_channels, 0.99))
    im = axes[0].imshow(nmse_channels, aspect="auto", cmap="magma", origin="lower",
                        vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axes[0], label="Per-channel NMSE (%) ; clipped at p99")
    axes[0].set_ylabel("Denoising step")
    axes[0].set_title("Output-channel error pattern (96 video latent-patch channels)")
    axes[1].plot(np.arange(96), ratio, color=GOLD, lw=1.3)
    axes[1].axhline(1, color="#444444", lw=1, ls="--")
    axes[1].set_xlabel("Output channel")
    axes[1].set_ylabel("Error share / BF16 energy share")
    save(fig, args.output_dir / "channel_error_pattern.png")

    frame_nmse = 100 * arrays["frame_error_energy"] / np.maximum(arrays["frame_signal_energy"], 1e-30)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    im = ax.imshow(frame_nmse, aspect="auto", origin="lower", cmap="magma",
                   vmin=0, vmax=float(np.quantile(frame_nmse, 0.99)))
    fig.colorbar(im, ax=ax, label="Video output NMSE per latent frame (%) ; clipped at p99")
    ax.set_xlabel("Latent video frame (0–36)")
    ax.set_ylabel("Denoising step (0–19)")
    ax.set_title("Where in video time is paired DiT output error largest?")
    save(fig, args.output_dir / "step_by_latent_frame_nmse.png")

    selected = (0, 10, 19)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), layout="constrained")
    for col, step in enumerate(selected):
        for row, (field, title, palette) in enumerate((
            ("token_error_rms", "Error RMS", "magma"),
            ("token_ref_rms", "BF16 output RMS", "viridis"))):
            field_map = arrays[f"step{step:02d}_{field}"][18]
            limit = float(np.quantile(field_map, 0.99))
            im = axes[row, col].imshow(field_map, cmap=palette, origin="lower", vmin=0, vmax=limit)
            axes[row, col].set_title(f"Step {step}, latent frame 18 · {title}", fontsize=10)
            axes[row, col].set_xlabel("Latent width patch (0–31)")
            axes[row, col].set_ylabel("Latent height patch (0–17)")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.suptitle("Spatial token maps; each panel has its own p99 color limit", fontsize=13)
    save(fig, args.output_dir / "spatial_token_maps.png")

    top_channel = summary["signed_spatial_map_top_channel"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), layout="constrained")
    for ax, step in zip(axes, selected, strict=True):
        data = arrays[f"step{step:02d}_signed_error_top_channel_frame18"]
        limit = float(np.quantile(np.abs(data), 0.99))
        im = ax.imshow(data, cmap="RdBu_r", origin="lower", vmin=-limit, vmax=limit)
        ax.set_title(f"Step {step}, channel {top_channel}")
        ax.set_xlabel("Latent width patch")
        ax.set_ylabel("Latent height patch")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Signed output error · latent frame 18 · independent symmetric p99 limits")
    save(fig, args.output_dir / "signed_top_channel_spatial_map.png")

    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    positives = [arrays[f"step{step:02d}_abs_error_over_ref_rms"] for step in selected]
    minimum_positive = min(float(values[values > 0].min()) for values in positives)
    maximum = max(float(values.max()) for values in positives)
    bins = np.geomspace(max(minimum_positive, 1e-8), maximum * 1.001, 90)
    for step, color in zip(selected, (BLUE, GOLD, ORANGE), strict=True):
        values = arrays[f"step{step:02d}_abs_error_over_ref_rms"]
        ax.hist(values[values > 0], bins=bins, histtype="step", density=True,
                linewidth=1.8, color=color, label=f"Step {step}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|quantized − BF16| / BF16 output RMS")
    ax.set_ylabel("Probability density (log scale)")
    ax.set_title("Absolute output-error distribution · 150k sampled elements per step")
    ax.legend(frameon=False)
    save(fig, args.output_dir / "error_magnitude_distribution.png")


if __name__ == "__main__":
    main()
