#!/usr/bin/env python3
"""Compare full-W4A4 vs keep-blocks-18-24-BF16 first-step error propagation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/quant_error_diagnosis")
PROP = ROOT / "block_propagation"
TAG = "keep_bf16_b18_24"
N_BLOCKS = 30


def load_sparse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    rows = json.loads(path.read_text())
    by_block: dict[int, list[dict]] = {}
    for row in rows:
        by_block.setdefault(int(row["block"]), []).append(row)
    blocks = sorted(by_block)
    x = np.asarray([b / (N_BLOCKS - 1) for b in blocks], dtype=float)
    mse = np.asarray(
        [np.mean([r["mse"] for r in by_block[b]]) for b in blocks], dtype=float
    )
    nmse = np.asarray(
        [np.mean([r["nmse"] for r in by_block[b]]) for b in blocks], dtype=float
    )
    return x, mse, nmse, blocks


def mean_stage(rows: list[dict], stage: str) -> tuple[float, float]:
    selected = [r for r in rows if r["stage"] == stage]
    return (
        float(np.mean([r["mse"] for r in selected])),
        float(np.mean([r["nmse"] for r in selected])),
    )


def main() -> None:
    base_sparse = load_sparse(PROP / "wan_firststep_metrics.json")
    keep_sparse = load_sparse(PROP / f"wan_firststep_metrics_{TAG}.json")
    base_dense = json.loads(
        (PROP / "wan_firststep_dense_head_metrics.json").read_text()
    )
    keep_dense = json.loads(
        (PROP / f"wan_firststep_dense_head_metrics_{TAG}.json").read_text()
    )
    keep_summary = json.loads((PROP / f"wan_{TAG}_summary.json").read_text())
    base_summary = {
        "proj_out": mean_stage(base_dense, "proj_out"),
        "cfg": next(
            r for r in base_dense if r["stage"] == "proj_out_cfg6_guided"
        ),
    }
    branch_base = [r for r in base_dense if r["stage"] == "proj_out"]
    base_amp_mse = base_summary["cfg"]["mse"] / float(
        np.mean([r["mse"] for r in branch_base])
    )
    base_amp_nmse = base_summary["cfg"]["nmse"] / float(
        np.mean([r["nmse"] for r in branch_base])
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)

    # MSE vs depth
    ax = axes[0, 0]
    ax.plot(base_sparse[0], base_sparse[1], marker="o", color="#F94144", label="full W4A4")
    ax.plot(
        keep_sparse[0],
        keep_sparse[1],
        marker="s",
        color="#277DA1",
        label="b18–24 BF16",
    )
    ax.axvspan(18 / 29, 24 / 29, color="#90BE6D", alpha=0.15, label="kept BF16")
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("MSE")
    ax.set_title("Wan first-step block MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # NMSE vs depth
    ax = axes[0, 1]
    ax.plot(
        base_sparse[0],
        100 * base_sparse[2],
        marker="o",
        color="#F94144",
        label="full W4A4",
    )
    ax.plot(
        keep_sparse[0],
        100 * keep_sparse[2],
        marker="s",
        color="#277DA1",
        label="b18–24 BF16",
    )
    ax.axvspan(18 / 29, 24 / 29, color="#90BE6D", alpha=0.15)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel("NMSE (%)")
    ax.set_title("Wan first-step block NMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # Output head
    stages = (
        "block29",
        "norm_out",
        "post_timestep_modulation",
        "proj_out",
    )
    labels = ("block29", "norm_out", "modulation", "proj_out")
    base_mse = [mean_stage(base_dense, s)[0] for s in stages]
    keep_mse = [mean_stage(keep_dense, s)[0] for s in stages]
    base_nmse = [100 * mean_stage(base_dense, s)[1] for s in stages]
    keep_nmse = [100 * mean_stage(keep_dense, s)[1] for s in stages]
    xs = np.arange(len(stages))
    ax = axes[1, 0]
    ax.plot(xs, base_mse, marker="o", color="#F94144", label="full W4A4 MSE")
    ax.plot(xs, keep_mse, marker="s", color="#277DA1", label="b18–24 BF16 MSE")
    ax.set_xticks(xs, labels, rotation=12)
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Output head MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(xs, base_nmse, marker="^", ls="--", color="#F9C74F", label="full NMSE")
    ax2.plot(xs, keep_nmse, marker="v", ls="--", color="#577590", label="keep NMSE")
    ax2.set_ylabel("NMSE (%)")
    ax2.set_yscale("log")

    # CFG bar comparison
    ax = axes[1, 1]
    cats = ["proj_out\n(branch avg)", "CFG=6\nguided"]
    base_vals_mse = [
        mean_stage(base_dense, "proj_out")[0],
        base_summary["cfg"]["mse"],
    ]
    keep_vals_mse = [
        mean_stage(keep_dense, "proj_out")[0],
        keep_summary["proj_out_cfg6_guided"]["mse"],
    ]
    base_vals_nmse = [
        100 * mean_stage(base_dense, "proj_out")[1],
        100 * base_summary["cfg"]["nmse"],
    ]
    keep_vals_nmse = [
        100 * mean_stage(keep_dense, "proj_out")[1],
        100 * keep_summary["proj_out_cfg6_guided"]["nmse"],
    ]
    xpos = np.arange(2)
    w = 0.35
    ax.bar(xpos - w / 2, base_vals_mse, w, color="#F94144", label="full W4A4 MSE")
    ax.bar(xpos + w / 2, keep_vals_mse, w, color="#277DA1", label="b18–24 BF16 MSE")
    ax.set_xticks(xpos, cats)
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("proj_out vs CFG=6 guided")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    axn = ax.twinx()
    axn.plot(xpos, base_vals_nmse, "o--", color="#F9C74F", label="full NMSE%")
    axn.plot(xpos, keep_vals_nmse, "s--", color="#577590", label="keep NMSE%")
    axn.set_ylabel("NMSE (%)")
    axn.set_yscale("log")

    fig.suptitle(
        "Wan first-step: full fake-W4A4 vs keep blocks 18–24 in BF16\n"
        f"CFG amplify MSE {base_amp_mse:.1f}×→{keep_summary['cfg_amplify_mse']:.1f}×, "
        f"NMSE {base_amp_nmse:.1f}×→{keep_summary['cfg_amplify_nmse']:.1f}×",
        fontsize=13,
    )
    out = ROOT / f"wan_firststep_{TAG}_vs_full.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"saved {out}")

    report = {
        "baseline_full_w4a4": {
            "proj_out_mse": base_vals_mse[0],
            "proj_out_nmse": base_vals_nmse[0] / 100,
            "cfg6_mse": base_vals_mse[1],
            "cfg6_nmse": base_vals_nmse[1] / 100,
            "cfg_amplify_mse": base_amp_mse,
            "cfg_amplify_nmse": base_amp_nmse,
            "sparse_peak_mse": float(base_sparse[1].max()),
            "sparse_peak_nmse": float(base_sparse[2].max()),
        },
        "keep_bf16_b18_24": {
            "proj_out_mse": keep_vals_mse[0],
            "proj_out_nmse": keep_vals_nmse[0] / 100,
            "cfg6_mse": keep_vals_mse[1],
            "cfg6_nmse": keep_vals_nmse[1] / 100,
            "cfg_amplify_mse": keep_summary["cfg_amplify_mse"],
            "cfg_amplify_nmse": keep_summary["cfg_amplify_nmse"],
            "sparse_peak_mse": float(keep_sparse[1].max()),
            "sparse_peak_nmse": float(keep_sparse[2].max()),
            "block29": keep_summary["block29"],
        },
        "figure": str(out),
    }
    (ROOT / f"wan_{TAG}_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
