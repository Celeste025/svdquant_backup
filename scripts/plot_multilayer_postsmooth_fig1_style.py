#!/usr/bin/env python3
"""Fig.1-style plots for multilayer postsmooth Linear-INPUT surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from plot_multilayer_fig1_style import DEPTHS, adjacent_similarity


ROOT = Path("outputs/timestep_outliers_multilayer")
OUT = ROOT / "fig1_style_postsmooth"
TYPES = {
    "attention_q": "qkv_proj (to_q) postsmooth INPUT",
    "attention_out": "o_proj postsmooth INPUT",
    "ffn_up": "ffn_up postsmooth INPUT",
    "ffn_down": "ffn_down postsmooth INPUT",
}


def load(model: str) -> dict:
    return torch.load(
        ROOT / f"{model}_multilayer_postsmooth_surfaces.pt",
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


def draw_curves(ax, surface: np.ndarray) -> None:
    raw, profile, jaccard = adjacent_similarity(surface)
    steps = np.arange(1, surface.shape[0])
    ax.plot(steps, raw, label="mean all-channel amplitude similarity", linewidth=2)
    ax.plot(
        steps, profile, label="mean relative-outlier profile similarity", linewidth=2
    )
    ax.plot(steps, jaccard, label="top-1% channel Jaccard", linewidth=1.6, alpha=0.85)
    ax.set_xlabel("later denoising step in adjacent pair")
    ax.set_ylabel("adjacent-timestep similarity")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()


def plot_heatmaps(payloads: dict[str, dict]) -> None:
    type_order = list(TYPES)
    for model, payload in payloads.items():
        surfaces = payload["surfaces"]
        layers = payload["layers"]
        blocks = sorted({v["block"] for v in layers.values()})
        all_values = torch.cat([x.flatten() for x in surfaces.values()])
        vmin = 0.0
        vmax = float(torch.quantile(all_values.clamp_min(0), 0.995))
        fig, axes = plt.subplots(
            len(type_order), len(blocks), figsize=(15, 11), constrained_layout=True
        )
        image = None
        for row, layer_type in enumerate(type_order):
            for col, block in enumerate(blocks):
                key = f"{layer_type}_b{block}"
                surface = surfaces[key].float().clamp_min(0)
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
                short = TYPES[layer_type].split(" postsmooth")[0]
                axes[row, col].set_title(f"{short} — block {block}")
                axes[row, col].set_xlabel("Denoising step")
                axes[row, col].set_ylabel("Input channel")
        title = "Wan2.1-T2V-1.3B" if model == "wan" else "FLUX.1-dev"
        fig.suptitle(
            f"{title}: postsmooth Linear-INPUT outliers (vmin=0)\n"
            r"$\log_2(\max|(x+\mathrm{shift})/s|)$",
            fontsize=15,
        )
        fig.colorbar(image, ax=axes, shrink=0.75, label="log2(absmax)")
        fig.savefig(ROOT / f"{model}_multilayer_postsmooth_surfaces.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {"flux": load("flux"), "wan": load("wan")}
    plot_heatmaps(payloads)
    manifest = []

    for layer_type, type_label in TYPES.items():
        for depth_label, blocks in DEPTHS:
            selected = {}
            layer_names = {}
            shifts = {}
            for model in ("flux", "wan"):
                key = f"{layer_type}_b{blocks[model]}"
                selected[model] = payloads[model]["surfaces"][key].float().numpy()
                layer_names[model] = payloads[model]["layers"][key]["name"]
                shifts[model] = float(payloads[model]["layers"][key]["shift"])

            zmin = 0.0
            zmax = max(float(x.max()) for x in selected.values())
            fig = plt.figure(figsize=(16, 10))
            display = (("flux", "FLUX.1-dev"), ("wan", "Wan2.1-1.3B"))
            for column, (model, model_label) in enumerate(display):
                surface = selected[model]
                surface_ax = fig.add_subplot(2, 2, column + 1, projection="3d")
                draw_surface(surface_ax, surface, zmin, zmax)
                surface_ax.set_title(
                    f"{model_label}: {type_label}\n"
                    f"{layer_names[model]}  shift={shifts[model]:g}"
                )
                curve_ax = fig.add_subplot(2, 2, column + 3)
                draw_curves(curve_ax, surface)
                curve_ax.set_title(
                    f"{model_label}: adjacent-timestep outlier similarity"
                )

            fig.suptitle(
                f"Postsmooth Linear-INPUT outliers — {type_label}, "
                f"{depth_label} depth  (vmin clipped to 0)\n"
                r"$v_{t,c}=\log_2(\max_{batch,tokens,CFG}|(x+\mathrm{shift})/s|)$",
                fontsize=15,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stem = f"{layer_type}_{depth_label}_wan_flux_postsmooth_fig1"
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
                "definition": payloads["flux"]["definition"],
                "vmin": 0.0,
                "figures": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
