#!/usr/bin/env python3
"""Fig.1-style plots for multilayer Linear-INPUT variance (raw vs postsmooth)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from plot_multilayer_fig1_style import DEPTHS, adjacent_similarity


ROOT = Path("outputs/timestep_outliers_multilayer")
OUT_RAW = ROOT / "fig1_style_variance_raw"
OUT_PS = ROOT / "fig1_style_variance_postsmooth"
OUT_CMP = ROOT / "fig1_style_variance_raw_vs_postsmooth"
TYPES = {
    "attention_q": "qkv_proj (to_q)",
    "attention_out": "o_proj",
    "ffn_up": "ffn_up",
    "ffn_down": "ffn_down",
}


def load(model: str) -> dict:
    return torch.load(
        ROOT / f"{model}_multilayer_variance_surfaces.pt",
        map_location="cpu",
        weights_only=False,
    )


def color_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Robust scale for log2(var); drop artificial VAR_EPS floor (-40)."""
    stacked = np.concatenate([a.ravel() for a in arrays])
    stacked = stacked[stacked > -39.5]
    if stacked.size == 0:
        stacked = np.concatenate([a.ravel() for a in arrays])
    zmin = float(np.quantile(stacked, 0.005))
    zmax = float(np.quantile(stacked, 0.995))
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
    ax.set_zlabel("log2 var")
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
    ax.legend(fontsize=8)


def plot_pair_folder(
    out_dir: Path,
    payloads: dict[str, dict],
    surface_key: str,
    kind_label: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for layer_type, type_label in TYPES.items():
        for depth_label, blocks in DEPTHS:
            selected = {}
            layer_names = {}
            for model in ("flux", "wan"):
                key = f"{layer_type}_b{blocks[model]}"
                selected[model] = (
                    payloads[model][surface_key][key].float().numpy()
                )
                layer_names[model] = payloads[model]["layers"][key]["name"]

            zmin, zmax = color_limits(*selected.values())
            fig = plt.figure(figsize=(16, 10))
            for column, (model, model_label) in enumerate(
                (("flux", "FLUX.1-dev"), ("wan", "Wan2.1-1.3B"))
            ):
                surface = selected[model]
                ax = fig.add_subplot(2, 2, column + 1, projection="3d")
                draw_surface(ax, surface, zmin, zmax)
                ax.set_title(f"{model_label}: {type_label} {kind_label}\n{layer_names[model]}")
                curve_ax = fig.add_subplot(2, 2, column + 3)
                draw_curves(curve_ax, surface)
                curve_ax.set_title(f"{model_label}: adjacent-timestep similarity")

            fig.suptitle(
                f"Per-channel token variance — {type_label}, {depth_label}, "
                f"{kind_label} (robust p0.5–p99.5; floor excluded)\n"
                r"$v_{t,c}=\log_2(\mathrm{Var}_{tokens}[\cdot]_{t,c})$",
                fontsize=15,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stem = f"{layer_type}_{depth_label}_wan_flux_var_{kind_label}"
            png = out_dir / f"{stem}.png"
            fig.savefig(png, dpi=180)
            plt.close(fig)
            manifest.append(
                {
                    "layer_type": layer_type,
                    "depth": depth_label,
                    "kind": kind_label,
                    "flux_layer": layer_names["flux"],
                    "wan_layer": layer_names["wan"],
                    "png": str(png),
                }
            )
            print(f"saved {png}", flush=True)
    return manifest


def plot_raw_vs_postsmooth(payloads: dict[str, dict]) -> list[dict]:
    OUT_CMP.mkdir(parents=True, exist_ok=True)
    manifest = []
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
                        (
                            model,
                            kind,
                            payloads[model][sk][key].float().numpy(),
                        )
                    )

            zmin, zmax = color_limits(*(p[2] for p in panels))

            fig = plt.figure(figsize=(16, 12))
            model_labels = {"flux": "FLUX.1-dev", "wan": "Wan2.1-1.3B"}
            for idx, (model, kind, surface) in enumerate(panels):
                ax = fig.add_subplot(3, 2, idx + 1, projection="3d")
                draw_surface(ax, surface, zmin, zmax)
                ax.set_title(
                    f"{model_labels[model]}: {kind}\n{names[model]}", fontsize=10
                )

            for col, model in enumerate(("flux", "wan")):
                ax = fig.add_subplot(3, 2, 5 + col)
                for kind, sk, style in (
                    ("raw", "surfaces_raw", "-"),
                    ("postsmooth", "surfaces_postsmooth", "--"),
                ):
                    key = f"{layer_type}_b{blocks[model]}"
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
                ax.set_title(f"{model_labels[model]}: variance envelopes")
                ax.set_xlabel("denoising step")
                ax.set_ylabel("log2 var")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7, ncol=2, frameon=False)

            fig.suptitle(
                f"Raw vs postsmooth Linear-INPUT variance — {type_label}, "
                f"{depth_label} (shared robust scale)\n"
                r"$v=\log_2\mathrm{Var}_{tokens}(x)$ vs "
                r"$\log_2\mathrm{Var}_{tokens}((x+\mathrm{shift})/s)$",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stem = f"{layer_type}_{depth_label}_wan_flux_var_raw_vs_postsmooth"
            png = OUT_CMP / f"{stem}.png"
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
    return manifest


def plot_heatmaps(payloads: dict[str, dict]) -> None:
    type_order = list(TYPES)
    for model, payload in payloads.items():
        for kind, sk in (("raw", "surfaces_raw"), ("postsmooth", "surfaces_postsmooth")):
            surfaces = payload[sk]
            layers = payload["layers"]
            blocks = sorted({v["block"] for v in layers.values()})
            all_values = torch.cat([x.flatten() for x in surfaces.values()])
            finite = all_values[all_values > -39.5]
            if finite.numel() == 0:
                finite = all_values
            vmin = float(torch.quantile(finite, 0.005))
            vmax = float(torch.quantile(finite, 0.995))
            if vmax <= vmin:
                vmax = vmin + 1e-3
            fig, axes = plt.subplots(
                len(type_order), len(blocks), figsize=(15, 11), constrained_layout=True
            )
            image = None
            for row, layer_type in enumerate(type_order):
                for col, block in enumerate(blocks):
                    key = f"{layer_type}_b{block}"
                    surface = surfaces[key].float().clamp(vmin, vmax)
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
                    axes[row, col].set_title(f"{TYPES[layer_type]} — b{block}")
                    axes[row, col].set_xlabel("step")
                    axes[row, col].set_ylabel("channel")
            title = "Wan2.1-T2V-1.3B" if model == "wan" else "FLUX.1-dev"
            fig.suptitle(
                f"{title}: Linear-INPUT token variance ({kind}, robust scale)\n"
                r"$\log_2\mathrm{Var}_{tokens}$",
                fontsize=15,
            )
            fig.colorbar(image, ax=axes, shrink=0.75, label="log2(var)")
            fig.savefig(
                ROOT / f"{model}_multilayer_variance_{kind}_surfaces.png", dpi=180
            )
            plt.close(fig)


def main() -> None:
    payloads = {"flux": load("flux"), "wan": load("wan")}
    plot_heatmaps(payloads)
    man_raw = plot_pair_folder(
        OUT_RAW, payloads, "surfaces_raw", "raw"
    )
    man_ps = plot_pair_folder(
        OUT_PS, payloads, "surfaces_postsmooth", "postsmooth"
    )
    man_cmp = plot_raw_vs_postsmooth(payloads)
    (ROOT / "variance_fig1_manifest.json").write_text(
        json.dumps(
            {
                "definition": payloads["flux"]["definition"],
                "color_scale": "robust p0.5–p99.5; exclude log2(VAR_EPS)=-40 floor",
                "raw": man_raw,
                "postsmooth": man_ps,
                "raw_vs_postsmooth": man_cmp,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
