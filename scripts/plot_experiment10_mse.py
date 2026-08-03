#!/usr/bin/env python3
"""Re-plot experiment 10 diagnostics using raw MSE (no energy normalization)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/quant_error_diagnosis")
OUT = ROOT / "mse_plots"
TYPES = ("attention_q", "attention_out", "ffn_up", "ffn_down")
LABELS = {
    "attention_q": "Attention Q/QKV",
    "attention_out": "Attention output",
    "ffn_up": "FFN up",
    "ffn_down": "FFN down",
}
COLORS = {"flux": "#277DA1", "wan": "#F94144"}


def plot_weights() -> None:
    payloads = {
        model: json.loads((ROOT / f"{model}_weight_reconstruction.json").read_text())
        for model in ("flux", "wan")
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for ax, kind in zip(axes, TYPES):
        values = {}
        for model, rows in payloads.items():
            chosen = [r for r in rows if r["key"].startswith(kind)]
            values[model] = [r["w4_plus_rank32_vs_ws"]["mse"] for r in chosen]
        x = np.arange(3)
        ax.bar(x - 0.18, values["flux"], 0.36, color=COLORS["flux"], label="FLUX-dev")
        ax.bar(x + 0.18, values["wan"], 0.36, color=COLORS["wan"], label="Wan")
        ax.set_xticks(x, ("shallow", "middle", "deep"))
        ax.set_title(LABELS[kind])
        ax.set_ylabel(r"weight reconstruction MSE  $\mathrm{mean}((Q-W_s)^2)$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(
        r"$W_s$ vs $Q_4(W_s-UD)+UD$  — raw MSE (no $/\,\mathrm{mean}(W_s^2)$)"
    )
    fig.savefig(OUT / "weight_reconstruction_mse.png", dpi=200)
    plt.close(fig)


def _wan_block_series(metric: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    wan = json.loads((ROOT / "block_propagation/wan_firststep_metrics.json").read_text())
    wan_dense = json.loads(
        (ROOT / "block_propagation/wan_firststep_dense_head_metrics.json").read_text()
    )
    by_block: dict[int, list[float]] = {}
    for row in wan:
        by_block.setdefault(row["block"], []).append(row[metric])
    for row in wan_dense:
        if row["stage"].startswith("block"):
            block = int(row["stage"].removeprefix("block"))
            by_block[block] = [
                value[metric] for value in wan_dense if value["stage"] == row["stage"]
            ]
    blocks = sorted(by_block)
    x = np.asarray(blocks) / 29.0
    y = np.asarray([np.mean(by_block[b]) for b in blocks])
    return x, y, blocks


def _flux_image_series(metric: str) -> list[tuple[float, float, str]]:
    flux = json.loads(
        (ROOT / "block_propagation/flux_firststep_metrics.json").read_text()
    )
    points = []
    for row in flux:
        if row["stage"] not in ("dual", "single") or row["stream"] != "image":
            continue
        if row["stage"] == "dual":
            depth = row["block"] / 56
            label = f"dual {row['block']}"
        else:
            depth = (19 + row["block"]) / 56
            label = f"single {row['block']}"
        points.append((depth, row[metric], label))
    points.sort()
    return points


def plot_blocks() -> None:
    wan_x, wan_mse, wan_blocks = _wan_block_series("mse")
    _, wan_nmse, _ = _wan_block_series("nmse")
    flux_mse = _flux_image_series("mse")
    flux_nmse = _flux_image_series("nmse")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    # Left: Wan MSE vs NMSE (twin y) — same model, fair comparison of metric choice
    ax = axes[0]
    ax.plot(
        wan_x,
        wan_mse,
        marker="o",
        color=COLORS["wan"],
        lw=2,
        label="MSE",
    )
    peak_i = int(np.argmax(wan_mse))
    ax.annotate(
        f"peak MSE @ block {wan_blocks[peak_i]}\n{wan_mse[peak_i]:.3e}",
        xy=(wan_x[peak_i], wan_mse[peak_i]),
        xytext=(12, 18),
        textcoords="offset points",
        color=COLORS["wan"],
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLORS["wan"], lw=1),
    )
    ax.set_xlabel("Relative transformer depth")
    ax.set_ylabel(r"Cumulative residual-stream MSE  $\mathrm{mean}(e^2)$")
    ax.set_title("Wan first-step block error (raw MSE)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    ax2 = ax.twinx()
    ax2.plot(
        wan_x,
        100 * wan_nmse,
        marker="s",
        ms=4,
        color="#F9C74F",
        lw=1.6,
        ls="--",
        label="NMSE (%)",
    )
    peak_n = int(np.argmax(wan_nmse))
    ax2.annotate(
        f"peak NMSE @ block {wan_blocks[peak_n]}\n{100 * wan_nmse[peak_n]:.2f}%",
        xy=(wan_x[peak_n], 100 * wan_nmse[peak_n]),
        xytext=(-8, -36),
        textcoords="offset points",
        ha="right",
        color="#B08900",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#B08900", lw=1),
    )
    ax2.set_ylabel("NMSE (%)", color="#B08900")
    ax2.tick_params(axis="y", labelcolor="#B08900")
    ax2.legend(loc="upper right", frameon=False)

    # Right: FLUX alone on MSE (scale incompatible with Wan)
    ax = axes[1]
    ax.plot(
        [x for x, _, _ in flux_mse],
        [y for _, y, _ in flux_mse],
        marker="o",
        color=COLORS["flux"],
        lw=2,
        label="MSE",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Relative transformer depth")
    ax.set_ylabel(r"Cumulative residual-stream MSE  $\mathrm{mean}(e^2)$")
    ax.set_title("FLUX-dev first-step block error (raw MSE, log)")
    ax.grid(alpha=0.25, which="both")
    ax.annotate(
        "Absolute MSE not comparable to Wan\n(activation scale differs)",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax_nm = ax.twinx()
    ax_nm.plot(
        [x for x, _, _ in flux_nmse],
        [100 * y for _, y, _ in flux_nmse],
        marker="s",
        ms=4,
        color="#90BE6D",
        lw=1.6,
        ls="--",
        label="NMSE (%)",
    )
    ax_nm.set_ylabel("NMSE (%)", color="#2D6A4F")
    ax_nm.tick_params(axis="y", labelcolor="#2D6A4F")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_nm.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")

    fig.suptitle(
        "Experiment 10.2: first-step block propagation — MSE vs NMSE",
        fontsize=13,
    )
    fig.savefig(OUT / "firststep_block_error_mse.png", dpi=220)
    plt.close(fig)


def plot_output_head() -> None:
    wan_dense = json.loads(
        (ROOT / "block_propagation/wan_firststep_dense_head_metrics.json").read_text()
    )
    head_stages = (
        "block29",
        "norm_out",
        "post_timestep_modulation",
        "proj_out",
    )
    head_labels = (
        "block 29\nresidual",
        "norm_out",
        "timestep\nmodulation",
        "proj_out",
    )
    mse_vals = [
        float(np.mean([r["mse"] for r in wan_dense if r["stage"] == stage]))
        for stage in head_stages
    ]
    nmse_vals = [
        float(np.mean([r["nmse"] for r in wan_dense if r["stage"] == stage]))
        for stage in head_stages
    ]
    powers = [
        float(np.mean([r["reference_rms"] ** 2 for r in wan_dense if r["stage"] == stage]))
        for stage in head_stages
    ]
    guided = next(r for r in wan_dense if r["stage"] == "proj_out_cfg6_guided")
    branch_mse = [
        r["mse"]
        for r in wan_dense
        if r["stage"] == "proj_out" and r["cfg_branch"] is not None
    ]
    branch_nmse = [
        r["nmse"]
        for r in wan_dense
        if r["stage"] == "proj_out" and r["cfg_branch"] is not None
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.15]},
    )

    # Panel 1: MSE along output head
    ax = axes[0]
    ax.plot(
        np.arange(len(mse_vals)),
        mse_vals,
        marker="o",
        lw=2.2,
        color=COLORS["wan"],
        label="MSE",
    )
    for i, v in enumerate(mse_vals):
        ax.annotate(
            f"{v:.2e}",
            xy=(i, v),
            xytext=(0, 8 if i < 2 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=COLORS["wan"],
        )
    ax.set_xticks(np.arange(len(mse_vals)), head_labels)
    ax.set_ylabel(r"MSE  $\mathrm{mean}((q-\mathrm{bf16})^2)$")
    ax.set_title("Output-head absolute error (MSE)")
    ax.grid(alpha=0.25)
    ax.set_yscale("log")

    # Panel 2: NMSE for contrast + ref power
    ax = axes[1]
    ax.plot(
        np.arange(len(nmse_vals)),
        100 * np.asarray(nmse_vals),
        marker="s",
        lw=2.2,
        color="#F9C74F",
        label="NMSE",
    )
    ax.set_xticks(np.arange(len(nmse_vals)), head_labels)
    ax.set_ylabel("NMSE (%)")
    ax.set_title("Same stages as NMSE (for contrast)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        np.arange(len(powers)),
        powers,
        marker="^",
        ms=5,
        lw=1.4,
        ls=":",
        color="#577590",
        label=r"$\mathrm{mean}(\mathrm{ref}^2)$",
    )
    ax2.set_ylabel(r"reference power  $\mathrm{mean}(\mathrm{bf16}^2)$", color="#577590")
    ax2.tick_params(axis="y", labelcolor="#577590")
    ax2.set_yscale("log")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)

    # Panel 3: CFG amplification under both metrics
    ax = axes[2]
    x = np.arange(3)
    labels = ("uncond", "cond", "CFG=6\nguided")
    mse_cfg = branch_mse + [guided["mse"]]
    nmse_cfg = branch_nmse + [guided["nmse"]]
    # Normalize each series to uncond=1 for amplify visualization, but also show absolute
    w = 0.36
    bars_m = ax.bar(
        x - w / 2,
        mse_cfg,
        w,
        color=COLORS["wan"],
        label="MSE",
    )
    ax.set_ylabel("MSE", color=COLORS["wan"])
    ax.tick_params(axis="y", labelcolor=COLORS["wan"])
    ax.set_yscale("log")
    ax2 = ax.twinx()
    bars_n = ax2.bar(
        x + w / 2,
        100 * np.asarray(nmse_cfg),
        w,
        color="#F9C74F",
        label="NMSE (%)",
    )
    ax2.set_ylabel("NMSE (%)", color="#B08900")
    ax2.tick_params(axis="y", labelcolor="#B08900")
    ax2.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_title("CFG error amplification")
    ax.grid(axis="y", alpha=0.25)
    amp_m = guided["mse"] / float(np.mean(branch_mse))
    amp_n = guided["nmse"] / float(np.mean(branch_nmse))
    ax.text(
        0.5,
        -0.22,
        f"amplify vs branch mean:  MSE ×{amp_m:.1f}   |   NMSE ×{amp_n:.1f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    for bar, val in zip(bars_m, mse_cfg):
        ax.annotate(
            f"{val:.2e}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["wan"],
            rotation=90,
        )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.suptitle(
        "Experiment 10.3: Wan output head — MSE removes the modulation “spike”",
        fontsize=13,
    )
    fig.savefig(OUT / "wan_firststep_output_head_mse.png", dpi=220)
    plt.close(fig)


def plot_mse_nmse_overview() -> None:
    """One-page overview: weight median/max + Wan focus blocks."""
    summary = json.loads((ROOT / "summary_mse.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # Weight median / max MSE
    ax = axes[0]
    models = ("flux", "wan")
    x = np.arange(2)
    med = [summary["weight_reconstruction"][m]["median_mse"] for m in models]
    mx = [summary["weight_reconstruction"][m]["max_mse"] for m in models]
    ax.bar(x - 0.18, med, 0.36, color=[COLORS[m] for m in models], label="median MSE")
    ax.bar(
        x + 0.18,
        mx,
        0.36,
        color=[COLORS[m] for m in models],
        alpha=0.45,
        hatch="//",
        label="max MSE",
    )
    ax.set_xticks(x, ("FLUX-dev", "Wan"))
    ax.set_ylabel("MSE")
    ax.set_title("10.1 Weight reconstruction (12 layers)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    # Focus blocks MSE bars
    ax = axes[1]
    focus = summary["wan_block_propagation_focus"]
    blocks = [int(b) for b in focus]
    mses = [focus[str(b)]["mse"] for b in blocks]
    colors = [
        "#F94144" if b == summary["wan_peak"]["by_mse"]["block"] else "#F9844A"
        for b in blocks
    ]
    ax.bar([str(b) for b in blocks], mses, color=colors)
    ax.set_xlabel("Block index")
    ax.set_ylabel("MSE (CFG-branch mean)")
    ax.set_title("10.2 Wan focus blocks — raw MSE")
    ax.grid(axis="y", alpha=0.25)
    peak = summary["wan_peak"]["by_mse"]["block"]
    ax.annotate(
        f"peak MSE @ {peak}",
        xy=(str(peak), focus[str(peak)]["mse"]),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["wan"],
    )

    fig.suptitle(
        r"Experiment 10 re-eval with MSE $=\,\mathrm{mean}((\mathrm{quant}-\mathrm{bf16})^2)$",
        fontsize=13,
    )
    fig.savefig(OUT / "experiment10_mse_overview.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_weights()
    plot_blocks()
    plot_output_head()
    plot_mse_nmse_overview()
    print("Wrote figures to", OUT.resolve())
    for path in sorted(OUT.glob("*.png")):
        print(" ", path.name)


if __name__ == "__main__":
    main()
