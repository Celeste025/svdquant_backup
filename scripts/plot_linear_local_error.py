#!/usr/bin/env python3
"""Plot independent per-Linear NMSE reports produced by scan_linear_nmse.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("wan2.1-1.3b", "rcm-wan2.1-1.3b", "flux.1-dev", "flux.1-schnell")
LABELS = {
    "wan2.1-1.3b": "Wan2.1-1.3B",
    "rcm-wan2.1-1.3b": "rCM-Wan2.1-1.3B",
    "flux.1-dev": "FLUX.1-dev",
    "flux.1-schnell": "FLUX.1-schnell",
}
COLORS = {
    "attention Q/K/V": "#e15759",
    "attention output": "#f28e2b",
    "FFN / MLP": "#59a14f",
    "condition / time": "#b07aa1",
    "other projection": "#4e79a7",
}


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    # Scan JSON is ordered by severity for inspection. Restore model order for a
    # layer-by-layer profile, retaining only successful forwards.
    return sorted((row for row in data["layers"] if "nmse" in row), key=lambda row: row["index"])


def layer_kind(name: str) -> str:
    """Assign a stable, architecture-agnostic colour category from module names."""
    lower = name.lower()
    if any(token in lower for token in ("to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj")):
        return "attention Q/K/V"
    if any(token in lower for token in ("to_out", "out_proj", "proj_out")) and "attn" in lower:
        return "attention output"
    if any(token in lower for token in ("ffn", ".ff.", ".mlp.", "net.")):
        return "FFN / MLP"
    if any(token in lower for token in ("embedder", "time_embed", "condition_embed", "context_embed")):
        return "condition / time"
    return "other projection"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "results" / "reports" / "linear_local_error_full_tokens",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results" / "reports" / "linear_local_error_full_tokens" / "per_linear_nmse.png",
    )
    args = parser.parse_args()
    args.out = args.out.resolve()

    reports = {}
    for model in ORDER:
        path = args.input_dir / f"{model}_linear_local_error.json"
        if path.is_file():
            reports[model] = load(path)
        else:
            print(f"[skip] report not available yet: {path}")
    if not reports:
        raise SystemExit("no scan reports found")

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, model in zip(axes.flat, ORDER):
        rows = reports.get(model)
        axis.set_title(LABELS[model])
        axis.set_xlabel("Linear layer index (model order)")
        axis.set_ylabel("independent NMSE (log scale)")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25, which="both")
        if not rows:
            axis.text(0.5, 0.5, "report pending", ha="center", va="center", transform=axis.transAxes)
            continue
        values = [row["nmse"] for row in rows]
        positions = list(range(1, len(rows) + 1))
        axis.plot(positions, values, color="#bab0ab", linewidth=0.55, zorder=1)
        for kind, color in COLORS.items():
            xy = [(index, row["nmse"]) for index, row in zip(positions, rows) if layer_kind(row["name"]) == kind]
            if xy:
                axis.scatter(*zip(*xy), s=13, color=color, label=kind, alpha=0.9, zorder=2)
        worst = max(rows, key=lambda row: row["nmse"])
        worst_x = rows.index(worst) + 1
        axis.scatter([worst_x], [worst["nmse"]], color="crimson", zorder=3)
        axis.annotate(
            f"max: {worst['name']}\n{worst['nmse']:.3g}",
            (worst_x, worst["nmse"]), xytext=(6, 6), textcoords="offset points", fontsize=7,
        )
        axis.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    figure.suptitle("Same-input independent Linear quantization error (all tokens)", fontsize=14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
