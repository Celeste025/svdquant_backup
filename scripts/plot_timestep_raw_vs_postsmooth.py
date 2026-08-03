#!/usr/bin/env python3
"""Plot raw vs postsmooth Linear-input outlier surfaces (Fig.1 style)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from plot_timestep_activation_outliers import metrics


ROOT = Path("outputs/timestep_outliers")


def load_pair(model: str) -> dict[str, np.ndarray]:
    payload = torch.load(
        ROOT / f"{model}_input_raw_vs_postsmooth.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {
        "raw": payload["raw_log2_absmax"].float().numpy(),
        "postsmooth": payload["postsmooth_log2_absmax"].float().numpy(),
        "layer": payload["layer"],
        "shift": float(payload["shift"]),
    }


def plot_surface(ax, surface: np.ndarray, *, vmin: float, vmax: float, title: str):
    steps = np.arange(surface.shape[0])
    channels = np.arange(surface.shape[1])
    step_grid, channel_grid = np.meshgrid(steps, channels)
    # Clip geometry to the color scale. Otherwise FLUX-raw near-zero channels at
    # log2(2**-20)=-20 dominate the Z axis and the real mass collapses to a
    # yellow-green roof (unlike Fig.1 OUTPUT surfaces, which never hit that floor).
    z = np.clip(surface.T, vmin, vmax)
    ax.plot_surface(
        channel_grid,
        step_grid,
        z,
        cmap=cm.viridis,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=False,
        rcount=min(surface.shape[1], 384),
        ccount=50,
    )
    ax.set_zlim(vmin, vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("channel index")
    ax.set_ylabel("denoising step")
    ax.set_zlabel("log2 absmax")
    ax.view_init(elev=24, azim=-62)


def robust_limits(*arrays: np.ndarray, lo: float = 0.005, hi: float = 0.995) -> tuple[float, float]:
    stacked = np.concatenate([a.ravel() for a in arrays])
    # Drop the artificial absmax floor before taking percentiles.
    floor = float(np.log2(2**-20))
    stacked = stacked[stacked > floor + 1e-6]
    if stacked.size == 0:
        stacked = np.concatenate([a.ravel() for a in arrays])
    vmin = float(np.quantile(stacked, lo))
    vmax = float(np.quantile(stacked, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    return vmin, vmax


def main() -> None:
    data = {
        "FLUX.1-dev": load_pair("flux"),
        "Wan2.1-1.3B": load_pair("wan"),
    }

    # Match Fig.1: ONE shared color/Z scale across panels so FLUX (higher) vs Wan
    # (lower) separate in viridis. Exclude the -20 absmax floor, then use robust
    # percentiles — plain global min/max on INPUT would still be ruined by rare
    # near-zero channels even after dropping exact -20.
    all_surfaces = [
        data[name][kind]
        for name in data
        for kind in ("raw", "postsmooth")
    ]
    vmin, vmax = robust_limits(*all_surfaces)

    fig = plt.figure(figsize=(16, 12))
    for row, (name, payload) in enumerate(data.items()):
        ax_raw = fig.add_subplot(3, 2, row * 2 + 1, projection="3d")
        ax_ps = fig.add_subplot(3, 2, row * 2 + 2, projection="3d")
        plot_surface(
            ax_raw,
            payload["raw"],
            vmin=vmin,
            vmax=vmax,
            title=f"{name}: raw Linear INPUT\n{payload['layer']}",
        )
        plot_surface(
            ax_ps,
            payload["postsmooth"],
            vmin=vmin,
            vmax=vmax,
            title=(
                f"{name}: postsmooth INPUT  (x+{payload['shift']:g})/s\n"
                f"{payload['layer']}"
            ),
        )

    for col, (name, payload) in enumerate(data.items()):
        ax = fig.add_subplot(3, 2, 5 + col)
        for kind, style in (("raw", "-"), ("postsmooth", "--")):
            surface = payload[kind]
            color_med, color_p99, color_max = ("#277DA1", "#F9C74F", "#F94144")
            ax.plot(
                np.median(surface, axis=1),
                style,
                color=color_med,
                lw=2,
                label=f"{kind} median",
            )
            ax.plot(
                np.quantile(surface, 0.99, axis=1),
                style,
                color=color_p99,
                lw=2,
                label=f"{kind} P99",
            )
            ax.plot(
                surface.max(axis=1),
                style,
                color=color_max,
                lw=1.8,
                label=f"{kind} max",
            )
        ax.set_title(f"{name}: timestep envelopes (solid=raw, dashed=postsmooth)")
        ax.set_xlabel("denoising step (0 = high noise)")
        ax.set_ylabel("log2 absmax")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2, frameon=False)

    fig.suptitle(
        "Same FFN-down layers: raw vs postsmooth Linear-input outliers\n"
        "(Fig.1 layers; shared robust color/Z scale p0.5–p99.5, floor log2(2**-20) excluded)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "wan_flux_timestep_outlier_raw_vs_postsmooth.png", dpi=180)
    plt.close(fig)

    # Postsmooth-only panel: same shared-scale + clip protocol as above (Fig.1 layout)
    fig = plt.figure(figsize=(16, 10))
    surfaces = {
        "FLUX.1-dev": data["FLUX.1-dev"]["postsmooth"],
        "Wan2.1-1.3B": data["Wan2.1-1.3B"]["postsmooth"],
    }
    zmin_p, zmax_p = robust_limits(*surfaces.values())
    for column, (name, surface) in enumerate(surfaces.items()):
        ax = fig.add_subplot(2, 2, column + 1, projection="3d")
        plot_surface(
            ax,
            surface,
            vmin=zmin_p,
            vmax=zmax_p,
            title=f"{name}: postsmooth FFN-down INPUT",
        )
        line_ax = fig.add_subplot(2, 2, column + 3)
        line_ax.plot(np.median(surface, axis=1), label="channel median", linewidth=2)
        line_ax.plot(
            np.quantile(surface, 0.99, axis=1), label="channel P99", linewidth=2
        )
        line_ax.plot(surface.max(axis=1), label="channel max", linewidth=2)
        line_ax.set_title(f"{name}: postsmooth timestep non-stationarity")
        line_ax.set_xlabel("denoising step (0 = high noise)")
        line_ax.set_ylabel("log2 absmax")
        line_ax.grid(alpha=0.25)
        line_ax.legend()
    fig.suptitle(
        "Timestep-dependent postsmooth activation outliers "
        "(same layers as Fig.1; Linear INPUT after (x+shift)/s)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "wan_flux_timestep_outlier_fig1_postsmooth.png", dpi=180)
    plt.close(fig)

    report = {
        "note": (
            "Original fig1 hooked layer OUTPUT. This comparison uses Linear INPUT "
            "before/after SmoothQuant scale on the same FFN-down modules."
        ),
        "layers": {name: data[name]["layer"] for name in data},
        "color_scale": {
            "shared_vmin_p005": vmin,
            "shared_vmax_p995": vmax,
            "postsmooth_only_vmin_p005": zmin_p,
            "postsmooth_only_vmax_p995": zmax_p,
        },
        "raw": {name: metrics(data[name]["raw"]) for name in data},
        "postsmooth": {name: metrics(data[name]["postsmooth"]) for name in data},
    }
    (ROOT / "timestep_outlier_raw_vs_postsmooth_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
