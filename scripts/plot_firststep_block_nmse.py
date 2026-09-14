#!/usr/bin/env python3
"""Plot first-step BF16 vs W4A4 per-layer NMSE line charts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("/ssd/2/wenjinqi.wjq")
DEFAULT_OUT = DATA_ROOT / "compare" / "firststep_block_nmse"


def plot_one(json_path: Path, out_png: Path, title: str) -> dict:
    meta = json.loads(json_path.read_text())
    rows = meta["rows"]
    xs = np.arange(len(rows))
    nmses = np.asarray([100.0 * r["nmse"] for r in rows], dtype=np.float64)
    labels = [r["name"].split("_", 1)[-1] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    ax.plot(xs, nmses, marker="o", markersize=3.5, linewidth=1.6, color="#1f4e79")
    ax.set_ylabel("NMSE (%)")
    ax.set_xlabel("Layer (model input → output)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)

    # sparse x tick labels
    step = max(1, len(labels) // 16)
    tick_idx = list(range(0, len(labels), step))
    if tick_idx[-1] != len(labels) - 1:
        tick_idx.append(len(labels) - 1)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=8)

    peak_i = int(np.argmax(nmses))
    ax.annotate(
        f"peak {nmses[peak_i]:.2f}%\n{labels[peak_i]}",
        xy=(xs[peak_i], nmses[peak_i]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#833"),
    )
    ax.annotate(
        f"final {nmses[-1]:.2f}%",
        xy=(xs[-1], nmses[-1]),
        xytext=(-50, 12),
        textcoords="offset points",
        fontsize=8,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    summary = {
        "json": str(json_path),
        "png": str(out_png),
        "n_layers": len(rows),
        "peak_name": rows[peak_i]["name"],
        "peak_nmse_pct": float(nmses[peak_i]),
        "final_name": rows[-1]["name"],
        "final_nmse_pct": float(nmses[-1]),
        "embed_nmse_pct": float(nmses[0]),
        "prompt": meta.get("prompt"),
        "seed": meta.get("seed"),
        "ckpt": meta.get("ckpt"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", choices=["wan", "flux", "both"], default="both")
    args = parser.parse_args()

    summaries = {}
    models = ["wan", "flux"] if args.model == "both" else [args.model]
    for m in models:
        jp = args.out_dir / f"{m}_firststep_block_nmse.json"
        if not jp.is_file():
            print(f"[skip] missing {jp}", flush=True)
            continue
        title = f"{m.upper()} first-step BF16 vs W4A4 layer NMSE"
        summaries[m] = plot_one(jp, args.out_dir / f"{m}_firststep_block_nmse.png", title)

    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
