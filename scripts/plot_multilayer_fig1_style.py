#!/usr/bin/env python3
"""Render every multi-layer probe in the original SVQ-GPTQ Fig.1-style layout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


ROOT = Path("outputs/timestep_outliers_multilayer")
OUT = ROOT / "fig1_style_similarity"
# User-facing names: qkv_proj / o_proj / ffn (up+down). Depths: shallow + deep
# (plus one mid-depth for coverage).
TYPES = {
    "attention_q": "qkv_proj (to_q)",
    "attention_out": "o_proj",
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
}
DEPTHS = (
    ("shallow", {"flux": 0, "wan": 0}),
    ("middle", {"flux": 9, "wan": 15}),
    ("deep", {"flux": 18, "wan": 29}),
)


def load(model: str) -> dict:
    return torch.load(
        ROOT / f"{model}_multilayer_surfaces.pt",
        map_location="cpu",
        weights_only=False,
    )


def draw_surface(ax, surface: np.ndarray, zmin: float, zmax: float) -> None:
    steps = np.arange(surface.shape[0])
    channels = np.arange(surface.shape[1])
    step_grid, channel_grid = np.meshgrid(steps, channels)
    z = np.clip(surface.T, zmin, zmax)
    ax.plot_surface(
        channel_grid,
        step_grid,
        z,
        cmap=cm.viridis,
        vmin=zmin,
        vmax=zmax,
        linewidth=0,
        antialiased=False,
        rcount=min(surface.shape[1], 384),
        ccount=50,
    )
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("channel index")
    ax.set_ylabel("denoising step")
    ax.set_zlabel("log2 absmax")
    ax.view_init(elev=24, azim=-62)


def adjacent_similarity(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Since surface is log2(a), 2**(-abs(delta)) equals min(a,b)/max(a,b).
    raw = np.mean(np.exp2(-np.abs(np.diff(surface, axis=0))), axis=1)

    # Remove each timestep's global channel scale. This measures whether the
    # relative outlier profile changes, rather than whether all channels scale
    # up or down together.
    relative = surface - np.median(surface, axis=1, keepdims=True)
    profile = np.mean(np.exp2(-np.abs(np.diff(relative, axis=0))), axis=1)

    count = max(1, round(surface.shape[1] * 0.01))
    top_sets = [
        set(np.argpartition(row, -count)[-count:].tolist()) for row in surface
    ]
    jaccard = np.asarray(
        [
            len(a & b) / max(1, len(a | b))
            for a, b in zip(top_sets[:-1], top_sets[1:])
        ]
    )
    return raw, profile, jaccard


def draw_curves(ax, surface: np.ndarray) -> None:
    raw, profile, jaccard = adjacent_similarity(surface)
    # Point t denotes similarity between timestep t-1 and t.
    steps = np.arange(1, surface.shape[0])
    ax.plot(
        steps,
        raw,
        label="mean all-channel amplitude similarity",
        linewidth=2,
    )
    ax.plot(
        steps,
        profile,
        label="mean relative-outlier profile similarity",
        linewidth=2,
    )
    ax.plot(
        steps,
        jaccard,
        label="top-1% channel Jaccard",
        linewidth=1.6,
        alpha=0.85,
    )
    ax.set_xlabel("later denoising step in adjacent pair")
    ax.set_ylabel("adjacent-timestep similarity")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {"flux": load("flux"), "wan": load("wan")}
    manifest = []

    for layer_type, type_label in TYPES.items():
        for depth_label, blocks in DEPTHS:
            selected = {}
            layer_names = {}
            for model in ("flux", "wan"):
                key = f"{layer_type}_b{blocks[model]}"
                selected[model] = payloads[model]["surfaces"][key].float().numpy()
                layer_names[model] = payloads[model]["layers"][key]["name"]

            # Clip color/Z floor to ~0 so rare sub-unit absmax channels do not
            # pull viridis into a yellow roof (same idea as Fig.1 readability).
            zmin = 0.0
            zmax = max(float(x.max()) for x in selected.values())
            fig = plt.figure(figsize=(16, 10))
            display = (("flux", "FLUX.1-dev"), ("wan", "Wan2.1-1.3B"))
            for column, (model, model_label) in enumerate(display):
                surface = selected[model]
                surface_ax = fig.add_subplot(2, 2, column + 1, projection="3d")
                draw_surface(surface_ax, surface, zmin, zmax)
                surface_ax.set_title(
                    f"{model_label}: BF16 {type_label}\n{layer_names[model]}"
                )

                curve_ax = fig.add_subplot(2, 2, column + 3)
                draw_curves(curve_ax, surface)
                curve_ax.set_title(
                    f"{model_label}: adjacent-timestep outlier similarity"
                )

            fig.suptitle(
                f"Timestep-dependent activation outliers — {type_label}, "
                f"{depth_label} depth  (vmin clipped to 0)\n"
                r"$v_{t,c}=\log_2(\max_{batch,tokens,CFG}|Y_{t,c}|)$",
                fontsize=16,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stem = f"{layer_type}_{depth_label}_wan_flux_fig1"
            png = OUT / f"{stem}.png"
            fig.savefig(png, dpi=180)
            plt.close(fig)
            manifest.append(
                {
                    "layer_type": layer_type,
                    "depth": depth_label,
                    "flux_layer": layer_names["flux"],
                    "wan_layer": layer_names["wan"],
                    "png": str(png),
                }
            )
            print(f"saved {png}", flush=True)

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "cell_definition": (
                    "log2(absmax over batch, all tokens, and for Wan both CFG "
                    "branches, of each output channel)"
                ),
                "curve_definition": {
                    "all_channel_amplitude_similarity": (
                        "mean_c 2^(-abs(v[t+1,c]-v[t,c])); equivalently the "
                        "mean per-channel min/max amplitude ratio"
                    ),
                    "relative_outlier_profile_similarity": (
                        "same similarity after subtracting each timestep's "
                        "channel median in log2 space"
                    ),
                    "top1pct_jaccard": (
                        "Jaccard overlap of the highest 1% channel identities "
                        "between adjacent timesteps"
                    ),
                },
                "figures": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
