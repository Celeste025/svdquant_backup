#!/usr/bin/env python3
"""Render the SageAttention 8-case metric tables as PNG images."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "h3_sage_attention_similarity.json"
VR_SRC = ROOT / "results" / "h3_sage_attention_visionreward.jsonl"
OUT_DIR = Path("/home/admin/workspace/aop_lab/app_data/figures")

HEAD_BG, MEAN_BG, STD_BG = "#2c3e50", "#eaecee", "#fdebd0"
NEG, POS, DARK = "#c0392b", "#1e8449", "#1a252f"

VARIANT_LABEL = {
    "svdquant": "SVDQuant (sdpa)",
    "svdquant_sageattn2": "SVDQuant + SageAttn2++",
    "svdquant_sageattn3": "SVDQuant + SageAttn3",
}


def load_summaries() -> dict[str, dict]:
    data = json.loads(SRC.read_text())
    return {entry["label"]: entry for entry in data["summary"]}


def load_visionreward() -> dict[str, dict]:
    scores: dict[str, dict] = {}
    if not VR_SRC.exists():
        return scores
    for line in VR_SRC.open():
        row = json.loads(line)
        scores[(row["case_id"], row["variant"])] = row
    return scores


def visionreward_summary(scores: dict[str, dict]) -> dict[str, dict[str, float]]:
    variants = ("bf16", "svdquant", "svdquant_sageattn2", "svdquant_sageattn3")
    out: dict[str, dict[str, float]] = {}
    for variant in variants:
        values = [row["visionreward_weighted_score"] for row in scores.values() if row["variant"] == variant]
        all_mean = sum(values) / len(values)
        no_case0 = [row["visionreward_weighted_score"] for row in scores.values() if row["variant"] == variant and row["case_id"] != "vbench2_000"]
        out[variant] = {"all": all_mean, "excl_case0": sum(no_case0) / len(no_case0)}
    return out


def style_table(tbl, nrow: int, colors: list[list[str]], bold_rows: set[int], first_col_left: bool = True) -> None:
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(HEAD_BG)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.055)
            continue
        if row in bold_rows:
            cell.set_facecolor(MEAN_BG)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#fbfcfc" if row % 2 else "#f2f5f6")
        cell.get_text().set_color(colors[row - 1][col] if row <= len(colors) else DARK)
        if first_col_left and col == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.03


def render(path: Path, title: str, subtitle: str, headers: list[str], widths: list[float], cells: list[list[str]], colors: list[list[str]], bold_rows: set[int], figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, colWidths=widths, cellLoc="center", colLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    style_table(tbl, len(cells), colors, bold_rows)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.965, color=DARK)
    ax.set_title(subtitle, fontsize=10, color="#566573", pad=14)
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.01, right=0.99)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved to {path}")


def main() -> None:
    summaries = load_summaries()
    vr_scores = load_visionreward()
    vr_summary = visionreward_summary(vr_scores)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Table 1: per-case SVDQuant (sdpa) vs bf16 ----
    entry = summaries["svdquant|vs_bf16"]
    headers = ["Case", "Prompt dims", "PSNR (dB) \u2191", "LPIPS \u2193", "rel-L2 \u2193"]
    widths = [0.16, 0.24, 0.2, 0.2, 0.2]
    dims = {
        "vbench2_000": "Instance Preservation",
        "vbench2_001": "Dynamic Attribute",
        "vbench2_002": "Composition",
        "vbench2_003": "Dynamic Spatial Relationship",
        "vbench2_004": "Dynamic Spatial Relationship",
        "vbench2_005": "Motion Order Understanding",
        "vbench2_006": "Mechanics",
        "vbench2_007": "Complex Plot",
    }
    cells, colors = [], []
    for row in entry["per_case"]:
        cells.append([row["case"], dims[row["case"]], f"{row['psnr_db']:.2f}", f"{row['lpips_alex']:.4f}", f"{row['rel_l2']:.4f}"])
        colors.append([DARK, "#566573", DARK, DARK, DARK])
    mean_all = summaries["svdquant|vs_bf16"]
    mean_no0 = summaries["svdquant|vs_bf16|excl_case0"]
    cells.append(["Mean (8 cases)", "", f"{mean_all['psnr_db_mean']:.2f} ± {mean_all['psnr_db_std']:.2f}", f"{mean_all['lpips_mean']:.4f} ± {mean_all['lpips_std']:.4f}", f"{mean_all['rel_l2_mean']:.4f} ± {mean_all['rel_l2_std']:.4f}"])
    cells.append(["Mean (excl. case 000)", "", f"{mean_no0['psnr_db_mean']:.2f} ± {mean_no0['psnr_db_std']:.2f}", f"{mean_no0['lpips_mean']:.4f} ± {mean_no0['lpips_std']:.4f}", f"{mean_no0['rel_l2_mean']:.4f} ± {mean_no0['rel_l2_std']:.4f}"])
    colors.append([DARK] * 5)
    colors.append([DARK] * 5)
    render(
        OUT_DIR / "h3_svdquant_similarity_table.png",
        "SVDQuant (SDPA attention) vs BF16 — Per-Case Video Similarity",
        "PSNR pools all frames/pixels (data range 1) | LPIPS = AlexNet, per-frame mean | rel-L2 = ||pred−ref||₂ / ||ref||₂ | 8 VBench-2.0 cases",
        headers, widths, cells, colors, bold_rows={len(cells) - 1, len(cells)},
        figsize=(11.5, 7.0),
    )

    # ---- Table 2: variant summary vs bf16 ----
    headers = ["Variant", "VisionReward", "PSNR (dB) \u2191", "LPIPS \u2193", "rel-L2 \u2193"]
    widths = [0.26, 0.17, 0.21, 0.18, 0.18]
    variants = ("svdquant", "svdquant_sageattn2", "svdquant_sageattn3")
    cells, colors = [], []
    for variant in variants:
        key = f"{variant}|vs_bf16|excl_case0"
        entry = summaries[key]
        vr = vr_summary.get(variant, {}).get("excl_case0")
        cells.append([
            VARIANT_LABEL[variant],
            f"{vr:+.4f}" if vr is not None else "n/a",
            f"{entry['psnr_db_mean']:.2f} ± {entry['psnr_db_std']:.2f}",
            f"{entry['lpips_mean']:.4f} ± {entry['lpips_std']:.4f}",
            f"{entry['rel_l2_mean']:.4f} ± {entry['rel_l2_std']:.4f}",
        ])
        colors.append([DARK, POS if vr is not None and vr >= 0 else NEG, DARK, DARK, DARK])
    bf16_vr = vr_summary.get("bf16", {}).get("excl_case0")
    cells.append(["BF16 baseline", f"{bf16_vr:+.4f}" if bf16_vr is not None else "n/a", "—", "—", "—"])
    colors.append([DARK, DARK, DARK, DARK, DARK])
    render(
        OUT_DIR / "h3_sage_attention_variant_summary_table.png",
        "MiniMax-H3 × SVDQuant × SageAttention — Variant Summary (vs BF16)",
        "7 cases (case 000 excluded) | similarity metrics computed against the matched BF16 video | LPIPS = AlexNet per-frame mean",
        headers, widths, cells, colors, bold_rows={3},
        figsize=(11.5, 4.2),
    )

    # ---- Table 3: attention-isolated comparison (vs svdquant) ----
    headers = ["Attention backend", "PSNR (dB) \u2191", "LPIPS \u2193", "rel-L2 \u2193"]
    widths = [0.34, 0.24, 0.21, 0.21]
    cells, colors = [], []
    for variant, label in (("svdquant_sageattn2", "SageAttn2++ vs SVDQuant"), ("svdquant_sageattn3", "SageAttn3 vs SVDQuant")):
        entry = summaries[f"{variant}|vs_svdquant|excl_case0"]
        cells.append([label, f"{entry['psnr_db_mean']:.2f} ± {entry['psnr_db_std']:.2f}", f"{entry['lpips_mean']:.4f} ± {entry['lpips_std']:.4f}", f"{entry['rel_l2_mean']:.4f} ± {entry['rel_l2_std']:.4f}"])
        colors.append([DARK, DARK, DARK, DARK])
    render(
        OUT_DIR / "h3_sage_attention_isolated_table.png",
        "Attention Kernel Error Isolated (Sage vs SDPA, both under SVDQuant)",
        "7 cases (case 000 excluded) | candidates compared against the matched SVDQuant (SDPA) video | lower LPIPS / rel-L2, higher PSNR = closer to SDPA output",
        headers, widths, cells, colors, bold_rows=set(),
        figsize=(10.5, 3.4),
    )


if __name__ == "__main__":
    main()
