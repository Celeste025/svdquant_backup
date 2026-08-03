#!/usr/bin/env python3
"""Plot SVQ-GPTQ Fig.1-style timestep/channel activation outliers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


ROOT = Path("outputs/timestep_outliers")


def load(name: str) -> np.ndarray:
    payload = torch.load(ROOT / f"{name}_surface.pt", map_location="cpu", weights_only=False)
    return payload["log2_absmax"].float().numpy()


def topk_jaccard(a: np.ndarray, b: np.ndarray, fraction: float = 0.01) -> float:
    count = max(1, round(a.size * fraction))
    aa = set(np.argpartition(a, -count)[-count:].tolist())
    bb = set(np.argpartition(b, -count)[-count:].tolist())
    return len(aa & bb) / len(aa | bb)


def metrics(surface: np.ndarray) -> dict:
    channel_range = np.ptp(surface, axis=0)
    step_p99 = np.quantile(surface, 0.99, axis=1)
    step_max = surface.max(axis=1)
    consecutive = [
        topk_jaccard(surface[index], surface[index + 1])
        for index in range(surface.shape[0] - 1)
    ]
    raw = np.exp2(surface)
    early = np.median(raw[:5])
    late = np.median(raw[-5:])
    return {
        "channels": int(surface.shape[1]),
        "log2_absmax_global_min": float(surface.min()),
        "log2_absmax_global_max": float(surface.max()),
        "per_channel_timestep_range_median": float(np.median(channel_range)),
        "per_channel_timestep_range_p90": float(np.quantile(channel_range, 0.9)),
        "per_channel_timestep_range_max": float(channel_range.max()),
        "step_p99_range": float(np.ptp(step_p99)),
        "step_max_range": float(np.ptp(step_max)),
        "early_to_late_median_absmax_ratio": float(early / max(late, 1e-12)),
        "top1pct_consecutive_jaccard_mean": float(np.mean(consecutive)),
        "top1pct_first_last_jaccard": float(topk_jaccard(surface[0], surface[-1])),
    }


def main() -> None:
    surfaces = {"FLUX.1-dev": load("flux"), "Wan2.1-1.3B": load("wan")}
    zmin = 0.0
    zmax = max(float(value.max()) for value in surfaces.values())
    fig = plt.figure(figsize=(16, 10))

    for column, (name, surface) in enumerate(surfaces.items()):
        ax = fig.add_subplot(2, 2, column + 1, projection="3d")
        # Keep every channel but transpose to channel x timestep like Fig.1.
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
        ax.set_title(f"{name}: BF16 FFN-down activation")
        ax.set_xlabel("channel index")
        ax.set_ylabel("denoising step")
        ax.set_zlabel("log2 absmax")
        ax.view_init(elev=24, azim=-62)

        line_ax = fig.add_subplot(2, 2, column + 3)
        line_ax.plot(np.median(surface, axis=1), label="channel median", linewidth=2)
        line_ax.plot(np.quantile(surface, 0.99, axis=1), label="channel P99", linewidth=2)
        line_ax.plot(surface.max(axis=1), label="channel max", linewidth=2)
        line_ax.set_title(f"{name}: timestep non-stationarity")
        line_ax.set_xlabel("denoising step (0 = high noise)")
        line_ax.set_ylabel("log2 absmax")
        line_ax.grid(alpha=0.25)
        line_ax.legend()

    fig.suptitle(
        "Timestep-dependent activation outliers (SVQ-GPTQ Fig.1 protocol)",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "wan_flux_timestep_outlier_fig1.png", dpi=180)

    report = {
        "protocol": {
            "statistic": "per-channel absmax across all tokens, then log2",
            "trajectory": "full BF16 seed44 50-step denoising trajectory",
            "wan_layer": "blocks.20.ffn.net.2; max across conditional and unconditional CFG calls",
            "flux_layer": "transformer_blocks.12.ff.net.2; selected at the same ~2/3 relative depth of the dual-stream stage",
        },
        **{name: metrics(surface) for name, surface in surfaces.items()},
    }
    (ROOT / "timestep_outlier_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
