#!/usr/bin/env python3
"""Plot absmax Linear-INPUT raw vs postsmooth in the variance comparison layout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from plot_multilayer_fig1_style import DEPTHS


ROOT = Path("outputs/timestep_outliers_multilayer")
OUT = ROOT / "fig1_style_absmax_raw_vs_postsmooth"
TYPES = {
    "attention_q": "qkv_proj (to_q)",
    "attention_out": "o_proj",
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
}


def load(model: str) -> dict:
    return torch.load(
        ROOT / f"{model}_multilayer_absmax_input_surfaces.pt",
        map_location="cpu",
        weights_only=False,
    )


def color_limits(*arrays: np.ndarray) -> tuple[float, float]:
    # Match absmax readability: floor at 0, robust high end (drop log2(2**-20)=-20).
    stacked = np.concatenate([a.ravel() for a in arrays])
    stacked = stacked[stacked > -19.5]
    if stacked.size == 0:
        stacked = np.concatenate([a.ravel() for a in arrays])
    zmin = 0.0
    zmax = float(np.quantile(stacked, 0.995))
    if zmax <= zmin:
        zmax = float(stacked.max()) if stacked.size else zmin + 1e-3
        if zmax <= zmin:
            zmax = zmin + 1e-3
    return zmin, zmax


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {"flux": load("flux"), "wan": load("wan")}
    manifest = []
    model_labels = {"flux": "FLUX.1-dev", "wan": "Wan2.1-1.3B"}

    for layer_type, type_label in TYPES.items():
        for depth_label, blocks in DEPTHS:
            panels = []
            names = {}
            for model in ("flux", "wan"):
                key = f"{layer_type}_b{blocks[model]}"
                names[model] = payloads[model]["layers"][key]["name"]
                for kind, sk in (
                    ("raw", "surfaces_raw"),
                    ("postsmooth", "surfaces_postsmooth"),
                ):
                    panels.append(
                        (model, kind, payloads[model][sk][key].float().numpy())
                    )

            zmin, zmax = color_limits(*(p[2] for p in panels))
            fig = plt.figure(figsize=(16, 12))
            for idx, (model, kind, surface) in enumerate(panels):
                ax = fig.add_subplot(3, 2, idx + 1, projection="3d")
                draw_surface(ax, surface, zmin, zmax)
                ax.set_title(
                    f"{model_labels[model]}: {kind}\n{names[model]}", fontsize=10
                )

            for col, model in enumerate(("flux", "wan")):
                ax = fig.add_subplot(3, 2, 5 + col)
                key = f"{layer_type}_b{blocks[model]}"
                for kind, sk, style in (
                    ("raw", "surfaces_raw", "-"),
                    ("postsmooth", "surfaces_postsmooth", "--"),
                ):
                    surface = payloads[model][sk][key].float().numpy()
                    ax.plot(
                        np.median(surface, axis=1),
                        style,
                        color="#277DA1",
                        lw=2,
                        label=f"{kind} median",
                    )
                    ax.plot(
                        np.quantile(surface, 0.99, axis=1),
                        style,
                        color="#F9C74F",
                        lw=2,
                        label=f"{kind} P99",
                    )
                    ax.plot(
                        surface.max(axis=1),
                        style,
                        color="#F94144",
                        lw=1.8,
                        label=f"{kind} max",
                    )
                ax.set_title(f"{model_labels[model]}: absmax envelopes")
                ax.set_xlabel("denoising step")
                ax.set_ylabel("log2 absmax")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7, ncol=2, frameon=False)

            fig.suptitle(
                f"Raw vs postsmooth Linear-INPUT absmax — {type_label}, "
                f"{depth_label} (shared scale, vmin=0)\n"
                r"$v=\log_2\max|x|$ vs $\log_2\max|(x+\mathrm{shift})/s|$",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stem = f"{layer_type}_{depth_label}_wan_flux_absmax_raw_vs_postsmooth"
            png = OUT / f"{stem}.png"
            fig.savefig(png, dpi=180)
            plt.close(fig)
            manifest.append(
                {
                    "layer_type": layer_type,
                    "depth": depth_label,
                    "flux_layer": names["flux"],
                    "wan_layer": names["wan"],
                    "png": str(png),
                }
            )
            print(f"saved {png}", flush=True)

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "definition": payloads["flux"]["definition"],
                "layout": "same as fig1_style_variance_raw_vs_postsmooth",
                "color_scale": "vmin=0, vmax=shared p99.5 (floor -20 excluded)",
                "figures": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
