#!/usr/bin/env python3
"""Plot all-token, hook-aware independent Linear NMSE reports."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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


def layer_kind(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj")):
        return "attention Q/K/V"
    if "attn" in n and any(x in n for x in ("to_out", "out_proj", "proj_out")):
        return "attention output"
    if any(x in n for x in ("ffn", ".ff.", ".mlp.", "net.")):
        return "FFN / MLP"
    if any(x in n for x in ("embedder", "time_embed", "condition_embed", "context_embed")):
        return "condition / time"
    return "other projection"


def aggregate(path: Path) -> tuple[list[dict], dict]:
    report = json.loads(path.read_text())
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    for row in report["records"]:
        value = totals[row["layer"]]
        value[0] += float(row["sum_sq_err"])
        value[1] += float(row["sum_ref_sq"])
        value[2] += int(row["numel"])
    rows = [
        {"name": name, "nmse": err / max(ref, 1e-30), "numel": count, "kind": layer_kind(name)}
        for name, (err, ref, count) in totals.items()
    ]
    # Natural module-name order yields the architectural layer order.
    rows.sort(key=lambda row: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", row["name"])])
    return rows, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=ROOT / "results/reports/linear_online_corrected_4x10")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.input_dir = args.input_dir.resolve()
    out = (args.input_dir / "per_linear_nmse.png") if args.out is None else args.out.resolve()

    reports = {}
    for model in ORDER:
        path = args.input_dir / f"{model}_online_records.json"
        if path.exists():
            reports[model] = aggregate(path)
        else:
            print(f"[skip] {path}")
    if not reports:
        raise SystemExit("no corrected online reports found")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), constrained_layout=True)
    for ax, model in zip(axes.flat, ORDER, strict=True):
        ax.set_title(LABELS[model])
        ax.set_xlabel("Linear layer (architectural order)")
        ax.set_ylabel("independent NMSE")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25, which="both")
        if model not in reports:
            ax.text(.5, .5, "report pending", ha="center", va="center", transform=ax.transAxes)
            continue
        rows, report = reports[model]
        x = list(range(1, len(rows) + 1))
        y = [row["nmse"] for row in rows]
        ax.plot(x, y, color="#bab0ab", linewidth=.55, zorder=1)
        for category, color in COLORS.items():
            points = [(i, row["nmse"]) for i, row in zip(x, rows, strict=True) if row["kind"] == category]
            if points:
                px, py = zip(*points)
                ax.scatter(px, py, s=12, color=color, label=category, zorder=2)
        worst = max(enumerate(rows, start=1), key=lambda pair: pair[1]["nmse"])
        ax.annotate(f"max: {worst[1]['name']}\n{worst[1]['nmse']:.2g}", (worst[0], worst[1]["nmse"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=6.7)
        audit = report.get("context_audit", {})
        supported = sum(x.get("status") == "supported" for x in audit.values())
        ax.text(.99, .02, f"all tokens; {report['n_prompts']} prompts\n{len(report['selected_timestep_indices'])} timestep(s); {supported}/{len(audit)} supported",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=7)
        ax.legend(loc="upper left", fontsize=6.7, frameon=False, ncol=2)
    fig.suptitle("Hook-aware, same-input independent Linear quantization error", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
