#!/usr/bin/env python3
"""Plot Exp7/10 diagnostics with both MSE and NMSE."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/quant_error_diagnosis")
TYPES = ("attention_q", "attention_out", "ffn_up", "ffn_down")
LABELS = {
    "attention_q": "Attention Q/QKV",
    "attention_out": "Attention output",
    "ffn_up": "FFN up",
    "ffn_down": "FFN down",
}
COLORS = {"flux": "#277DA1", "wan": "#F94144"}


def aggregate_activation(model: str):
    rows = json.loads(
        (ROOT / "postsmooth_activation" / f"{model}_metrics.json").read_text()
    )
    grouped = {}
    for row in rows:
        key = (row["layer_type"], row["block"], row["step"])
        grouped.setdefault(key, []).append(row)
    result = []
    for (kind, block, step), values in grouped.items():
        result.append(
            {
                "type": kind,
                "block": block,
                "step": step,
                "mse": float(np.mean([v["mse"] for v in values])),
                "nmse": float(np.mean([v["nmse"] for v in values])),
                "max_over_rms": float(np.mean([v["max_over_rms"] for v in values])),
                "p999_over_rms": float(np.mean([v["p999_over_rms"] for v in values])),
            }
        )
    return result


def plot_activation(data):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for col, kind in enumerate(TYPES):
        for model in ("flux", "wan"):
            blocks = sorted({r["block"] for r in data[model] if r["type"] == kind})
            for index, block in enumerate(blocks):
                rows = sorted(
                    [r for r in data[model] if r["type"] == kind and r["block"] == block],
                    key=lambda x: x["step"],
                )
                label = f"{model.upper()} b{block}"
                style = ("-", "--", ":")[index]
                axes[0, col].plot(
                    [r["step"] for r in rows],
                    [r["mse"] for r in rows],
                    color=COLORS[model],
                    linestyle=style,
                    marker="o",
                    ms=3,
                    label=label,
                )
                axes[1, col].plot(
                    [r["step"] for r in rows],
                    [100 * r["nmse"] for r in rows],
                    color=COLORS[model],
                    linestyle=style,
                    marker="o",
                    ms=3,
                    label=label,
                )
        axes[0, col].set_title(LABELS[kind])
        axes[0, col].set_ylabel("A4 MSE")
        axes[0, col].set_yscale("log")
        axes[1, col].set_ylabel("A4 NMSE (%)")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Denoising step")
        for row in range(2):
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("Exp7 matched postsmooth A4 — MSE (top) / NMSE (bottom), group=64")
    fig.savefig(ROOT / "postsmooth_activation_a4_error.png", dpi=200)
    fig.savefig(ROOT / "postsmooth_activation_a4_error_mse_nmse.png", dpi=200)
    plt.close(fig)


def plot_weights():
    payloads = {
        model: json.loads((ROOT / f"{model}_weight_reconstruction.json").read_text())
        for model in ("flux", "wan")
    }
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    summary = {}
    for col, kind in enumerate(TYPES):
        mse_vals = {}
        nmse_vals = {}
        for model, rows in payloads.items():
            chosen = [r for r in rows if r["key"].startswith(kind)]
            mse_vals[model] = [r["w4_plus_rank32_vs_ws"]["mse"] for r in chosen]
            nmse_vals[model] = [100 * r["w4_plus_rank32_vs_ws"]["nmse"] for r in chosen]
        x = np.arange(3)
        axes[0, col].bar(
            x - 0.18, mse_vals["flux"], 0.36, color=COLORS["flux"], label="FLUX-dev"
        )
        axes[0, col].bar(
            x + 0.18, mse_vals["wan"], 0.36, color=COLORS["wan"], label="Wan"
        )
        axes[1, col].bar(
            x - 0.18, nmse_vals["flux"], 0.36, color=COLORS["flux"], label="FLUX-dev"
        )
        axes[1, col].bar(
            x + 0.18, nmse_vals["wan"], 0.36, color=COLORS["wan"], label="Wan"
        )
        for row in range(2):
            axes[row, col].set_xticks(x, ("shallow", "middle", "deep"))
            axes[row, col].grid(axis="y", alpha=0.25)
        axes[0, col].set_title(LABELS[kind])
        axes[0, col].set_ylabel("MSE")
        axes[0, col].ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
        axes[1, col].set_ylabel("NMSE (%)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(r"Exp10.1 $W_s$ vs $Q_4(W_s-UD)+UD$ — MSE (top) / NMSE (bottom)")
    fig.savefig(ROOT / "weight_reconstruction_error.png", dpi=200)
    fig.savefig(ROOT / "weight_reconstruction_error_mse_nmse.png", dpi=200)
    plt.close(fig)
    for model, rows in payloads.items():
        mses = [r["w4_plus_rank32_vs_ws"]["mse"] for r in rows]
        nmses = [r["w4_plus_rank32_vs_ws"]["nmse"] for r in rows]
        summary[model] = {
            "median_mse": float(np.median(mses)),
            "max_mse": float(max(mses)),
            "median_nmse": float(np.median(nmses)),
            "max_nmse": float(max(nmses)),
        }
    return summary


def _wan_block_metric(metric: str):
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
    return (
        np.asarray(blocks) / 29.0,
        np.asarray([np.mean(by_block[b]) for b in blocks]),
        blocks,
    )


def _flux_image_metric(metric: str):
    flux = json.loads(
        (ROOT / "block_propagation/flux_firststep_metrics.json").read_text()
    )
    points = []
    for row in flux:
        if row["stage"] not in ("dual", "single") or row.get("stream") != "image":
            continue
        if row["stage"] == "dual":
            depth = row["block"] / 56
        else:
            depth = (19 + row["block"]) / 56
        points.append((depth, row[metric]))
    points.sort()
    return points


def plot_blocks():
    wan_x, wan_mse, wan_blocks = _wan_block_metric("mse")
    _, wan_nmse, _ = _wan_block_metric("nmse")
    flux_mse = _flux_image_metric("mse")
    flux_nmse = _flux_image_metric("nmse")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)

    axes[0, 0].plot(wan_x, wan_mse, marker="o", color=COLORS["wan"], label="Wan MSE")
    peak_i = int(np.argmax(wan_mse))
    axes[0, 0].annotate(
        f"peak MSE @ b{wan_blocks[peak_i]}\n{wan_mse[peak_i]:.3e}",
        xy=(wan_x[peak_i], wan_mse[peak_i]),
        xytext=(10, 12),
        textcoords="offset points",
        color=COLORS["wan"],
        fontsize=9,
    )
    ax_t = axes[0, 0].twinx()
    ax_t.plot(
        wan_x,
        100 * wan_nmse,
        marker="s",
        ms=4,
        ls="--",
        color="#F9C74F",
        label="Wan NMSE",
    )
    peak_n = int(np.argmax(wan_nmse))
    ax_t.annotate(
        f"peak NMSE @ b{wan_blocks[peak_n]}\n{100 * wan_nmse[peak_n]:.2f}%",
        xy=(wan_x[peak_n], 100 * wan_nmse[peak_n]),
        xytext=(-8, -30),
        textcoords="offset points",
        ha="right",
        color="#B08900",
        fontsize=9,
    )
    axes[0, 0].set_ylabel("MSE")
    ax_t.set_ylabel("NMSE (%)", color="#B08900")
    axes[0, 0].set_title("Wan first-step block error")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].set_xlabel("Relative depth")

    axes[0, 1].plot(
        [x for x, _ in flux_mse],
        [y for _, y in flux_mse],
        marker="o",
        color=COLORS["flux"],
        label="FLUX MSE",
    )
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("FLUX image residual (MSE scale ≠ Wan)")
    axes[0, 1].grid(alpha=0.25)
    ax_f = axes[0, 1].twinx()
    ax_f.plot(
        [x for x, _ in flux_nmse],
        [100 * y for _, y in flux_nmse],
        marker="s",
        ms=4,
        ls="--",
        color="#90BE6D",
        label="FLUX NMSE",
    )
    ax_f.set_ylabel("NMSE (%)", color="#2D6A4F")
    axes[0, 1].set_xlabel("Relative depth")

    # Output head
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
        "block29",
        "norm_out",
        "modulation",
        "proj_out",
    )
    head_mse = [
        float(np.mean([r["mse"] for r in wan_dense if r["stage"] == s]))
        for s in head_stages
    ]
    head_nmse = [
        float(np.mean([r["nmse"] for r in wan_dense if r["stage"] == s]))
        for s in head_stages
    ]
    powers = [
        float(np.mean([r["reference_rms"] ** 2 for r in wan_dense if r["stage"] == s]))
        for s in head_stages
    ]
    axes[1, 0].plot(np.arange(4), head_mse, marker="o", color=COLORS["wan"], label="MSE")
    axes[1, 0].set_xticks(np.arange(4), head_labels, rotation=15)
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Wan output head")
    axes[1, 0].grid(alpha=0.25)
    ax_h = axes[1, 0].twinx()
    ax_h.plot(
        np.arange(4),
        100 * np.asarray(head_nmse),
        marker="s",
        ls="--",
        color="#F9C74F",
        label="NMSE",
    )
    ax_h.plot(
        np.arange(4),
        powers,
        marker="^",
        ls=":",
        color="#577590",
        label="ref power",
    )
    ax_h.set_ylabel("NMSE (%) / ref power")
    ax_h.set_yscale("log")

    # FLUX output head: last residual → LN → AdaLN modulation → proj_out
    # (AdaLayerNormContinuous split to mirror Wan's norm_out / modulation)
    flux = json.loads(
        (ROOT / "block_propagation/flux_firststep_metrics.json").read_text()
    )
    flux_last = next(
        r
        for r in flux
        if r["stage"] == "single" and r["block"] == 37 and r["stream"] == "image"
    )
    flux_norm = next(r for r in flux if r["stage"] == "norm_out")
    flux_mod = next(r for r in flux if r["stage"] == "post_timestep_modulation")
    flux_proj = next(r for r in flux if r["stage"] == "proj_out")
    flux_head_stages = ("last residual", "norm_out", "modulation", "proj_out")
    flux_head_mse = [
        flux_last["mse"],
        flux_norm["mse"],
        flux_mod["mse"],
        flux_proj["mse"],
    ]
    flux_head_nmse = [
        flux_last["nmse"],
        flux_norm["nmse"],
        flux_mod["nmse"],
        flux_proj["nmse"],
    ]
    flux_powers = [
        flux_last["reference_rms"] ** 2,
        flux_norm["reference_rms"] ** 2,
        flux_mod["reference_rms"] ** 2,
        flux_proj["reference_rms"] ** 2,
    ]
    axes[1, 1].plot(
        np.arange(4), flux_head_mse, marker="o", color=COLORS["flux"], label="MSE"
    )
    axes[1, 1].set_xticks(np.arange(4), flux_head_stages, rotation=15)
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("FLUX output head")
    axes[1, 1].grid(alpha=0.25)
    ax_fh = axes[1, 1].twinx()
    ax_fh.plot(
        np.arange(4),
        100 * np.asarray(flux_head_nmse),
        marker="s",
        ls="--",
        color="#90BE6D",
        label="NMSE",
    )
    ax_fh.plot(
        np.arange(4),
        flux_powers,
        marker="^",
        ls=":",
        color="#577590",
        label="ref power",
    )
    ax_fh.set_ylabel("NMSE (%) / ref power")
    ax_fh.set_yscale("log")

    guided = next(r for r in wan_dense if r["stage"] == "proj_out_cfg6_guided")
    branch = [
        r
        for r in wan_dense
        if r["stage"] == "proj_out" and r["cfg_branch"] is not None
    ]
    branch = sorted(branch, key=lambda r: r["cfg_branch"])
    amp_m = guided["mse"] / float(np.mean([branch[0]["mse"], branch[1]["mse"]]))
    amp_n = guided["nmse"] / float(np.mean([branch[0]["nmse"], branch[1]["nmse"]]))

    fig.suptitle("Exp10.2–10.3 first-step block / output-head — MSE + NMSE")
    fig.savefig(ROOT / "firststep_block_error_propagation.png", dpi=220)
    fig.savefig(ROOT / "firststep_block_error_mse_nmse.png", dpi=220)
    plt.close(fig)

    # Dedicated side-by-side Wan vs FLUX output-head figure
    fig, (ax_w, ax_f) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax_w.plot(np.arange(4), head_mse, marker="o", color=COLORS["wan"], label="MSE")
    ax_w2 = ax_w.twinx()
    ax_w2.plot(
        np.arange(4),
        100 * np.asarray(head_nmse),
        marker="s",
        ls="--",
        color="#F9C74F",
        label="NMSE",
    )
    ax_w2.plot(
        np.arange(4), powers, marker="^", ls=":", color="#577590", label="ref power"
    )
    ax_w.set_xticks(np.arange(4), head_labels, rotation=12)
    ax_w.set_ylabel("MSE")
    ax_w2.set_ylabel("NMSE (%) / ref power")
    ax_w.set_yscale("log")
    ax_w2.set_yscale("log")
    ax_w.set_title("Wan output-head error")
    ax_w.grid(alpha=0.25)

    ax_f.plot(
        np.arange(4), flux_head_mse, marker="o", color=COLORS["flux"], label="MSE"
    )
    ax_f2 = ax_f.twinx()
    ax_f2.plot(
        np.arange(4),
        100 * np.asarray(flux_head_nmse),
        marker="s",
        ls="--",
        color="#90BE6D",
        label="NMSE",
    )
    ax_f2.plot(
        np.arange(4),
        flux_powers,
        marker="^",
        ls=":",
        color="#577590",
        label="ref power",
    )
    ax_f.set_xticks(np.arange(4), flux_head_stages, rotation=12)
    ax_f.set_ylabel("MSE")
    ax_f2.set_ylabel("NMSE (%) / ref power")
    ax_f.set_yscale("log")
    ax_f2.set_yscale("log")
    ax_f.set_title("FLUX output-head error")
    ax_f.grid(alpha=0.25)
    fig.savefig(ROOT / "wan_firststep_output_head_error.png", dpi=220)
    fig.savefig(ROOT / "wan_firststep_output_head_error_mse_nmse.png", dpi=220)
    # also save a clearer dual name
    fig.savefig(ROOT / "output_head_wan_flux_mse_nmse.png", dpi=220)
    plt.close(fig)

    return {
        "wan_peak_mse": float(wan_mse.max()),
        "wan_peak_mse_block": int(wan_blocks[peak_i]),
        "wan_peak_nmse": float(wan_nmse.max()),
        "wan_peak_nmse_block": int(wan_blocks[peak_n]),
        "wan_final_sampled_block_mse": float(wan_mse[-1]),
        "wan_final_sampled_block_nmse": float(wan_nmse[-1]),
        "wan_dense_peak_block": int(wan_blocks[peak_i]),
        "wan_output_head": {
            stage: {"mse": float(m), "nmse": float(n), "ref_power": float(p)}
            for stage, m, n, p in zip(head_stages, head_mse, head_nmse, powers)
        },
        "flux_output_head": {
            stage: {"mse": float(m), "nmse": float(n), "ref_power": float(p)}
            for stage, m, n, p in zip(
                (
                    "last_residual",
                    "norm_out",
                    "post_timestep_modulation",
                    "proj_out",
                ),
                flux_head_mse,
                flux_head_nmse,
                flux_powers,
            )
        },
        "wan_proj_out_cfg6_guided_mse": float(guided["mse"]),
        "wan_proj_out_cfg6_guided_nmse": float(guided["nmse"]),
        "wan_cfg_amplify_mse": float(amp_m),
        "wan_cfg_amplify_nmse": float(amp_n),
        "flux_peak_mse": float(max(y for _, y in flux_mse)),
        "flux_peak_nmse": float(max(y for _, y in flux_nmse)),
        "flux_final_sampled_block_mse": float(flux_mse[-1][1]),
        "flux_final_sampled_block_nmse": float(flux_nmse[-1][1]),
        "note": "Cross-model comparisons should use NMSE; raw MSE scales differ.",
    }


def plot_evidence_chain(summary):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    cards = (
        (
            "1. Post-smooth A4",
            [
                100 * summary["activation"]["flux"]["median_a4_nmse"],
                100 * summary["activation"]["wan"]["median_a4_nmse"],
            ],
            "median NMSE (%)",
            "Wan is not worse",
        ),
        (
            "2. W4 + rank-32",
            [
                100 * summary["weight"]["flux"]["median_nmse"],
                100 * summary["weight"]["wan"]["median_nmse"],
            ],
            "median NMSE (%)",
            "Wan is locally better",
        ),
        (
            "3. First-step blocks",
            [
                100 * summary["block_propagation"]["flux_peak_nmse"],
                100 * summary["block_propagation"]["wan_peak_nmse"],
            ],
            "peak residual NMSE (%)",
            "Wan mid-blocks amplify",
        ),
        (
            "4. Wan CFG output",
            [
                100 * summary["block_propagation"]["wan_output_head"]["proj_out"]["nmse"],
                100 * summary["block_propagation"]["wan_proj_out_cfg6_guided_nmse"],
            ],
            "output NMSE (%)",
            "CFG=6 amplifies error",
        ),
    )
    for index, (title, values, ylabel, conclusion) in enumerate(cards):
        ax = axes[index]
        labels = ("FLUX", "Wan") if index < 3 else ("branch avg.", "CFG=6")
        colors = (
            (COLORS["flux"], COLORS["wan"]) if index < 3 else ("#90BE6D", COLORS["wan"])
        )
        bars = ax.bar(np.arange(2), values, color=colors, width=0.62)
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.text(
            0.5,
            -0.20,
            conclusion,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    fig.suptitle(
        "Four-stage diagnosis (NMSE): local quant error does not explain Wan quality loss",
        fontsize=14,
    )
    fig.savefig(ROOT / "four_experiment_evidence_chain.png", dpi=220)
    plt.close(fig)


def main():
    activation = {model: aggregate_activation(model) for model in ("flux", "wan")}
    plot_activation(activation)
    weight = plot_weights()
    block = plot_blocks()
    summary = {
        "weight": weight,
        "block_propagation": block,
        "wan_calibration_joint_linear_output": {
            "mean_final_nmse": 0.009113401612270384,
            "note": (
                "Mean calibration-set output NMSE after joint A4 input, W4 "
                "residual, and rank-32 compensation over 210 quantization groups."
            ),
        },
        "activation": {
            model: {
                "median_a4_mse": float(np.median([r["mse"] for r in rows])),
                "median_a4_nmse": float(np.median([r["nmse"] for r in rows])),
                "median_max_over_rms": float(
                    np.median([r["max_over_rms"] for r in rows])
                ),
            }
            for model, rows in activation.items()
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_evidence_chain(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
