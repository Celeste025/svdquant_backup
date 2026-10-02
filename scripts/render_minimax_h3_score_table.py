#!/usr/bin/env python3
"""Render the MiniMax-H3 x VBench-2.0 detailed score table as a PNG image."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = (
    "/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3"
    "/metadata/h3_vbench2_100/eval_results"
)
SRC = f"{RESULTS}/score_table_detailed.txt"
OUT = f"{RESULTS}/score_table.png"

HEADERS = ["Dimension", "Category", "N", "bf16", "w4a4", "SVDQuant r64", "Δ w4a4", "Δ SVDQuant"]
WIDTHS = [0.215, 0.155, 0.05, 0.105, 0.105, 0.125, 0.12, 0.125]

NEG, POS, ZERO = "#c0392b", "#1e8449", "#7f8c8d"
HEAD_BG, MEAN_BG = "#2c3e50", "#eaecee"


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def main() -> None:
    rows = []
    with open(SRC) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    for line in lines[1:]:
        p = line.split("|")
        if p[0] == "MEAN":
            continue
        rows.append(
            {
                "dim": p[0],
                "cat": p[1],
                "n": p[2],
                "bf16": float(p[3]),
                "w4a4": float(p[4]),
                "svd": float(p[5]),
                "dw": float(p[6]),
                "ds": float(p[7]),
            }
        )
    mean = lines[-1].split("|")

    cells, colors = [], []
    for r in rows:
        cells.append(
            [
                r["dim"],
                r["cat"],
                r["n"],
                f"{r['bf16']:.4f}",
                f"{r['w4a4']:.4f}",
                f"{r['svd']:.4f}",
                fmt(r["dw"]),
                fmt(r["ds"]),
            ]
        )
        c = ["#2c3e50"] * 6
        c.append(NEG if r["dw"] < -1e-9 else (POS if r["dw"] > 1e-9 else ZERO))
        c.append(NEG if r["ds"] < -1e-9 else (POS if r["ds"] > 1e-9 else ZERO))
        colors.append(c)

    mean_cells = [
        "MEAN",
        "",
        "",
        f"{float(mean[3]):.4f}",
        f"{float(mean[4]):.4f}",
        f"{float(mean[5]):.4f}",
        fmt(float(mean[6])),
        fmt(float(mean[7])),
    ]
    mean_colors = ["#2c3e50", "#2c3e50", "#2c3e50", "#2c3e50", "#2c3e50", "#2c3e50", NEG, NEG]

    nrow = len(cells) + 1
    fig, ax = plt.subplots(figsize=(14.5, 8.0), dpi=200)
    ax.axis("off")

    tbl = ax.table(
        cellText=cells + [mean_cells],
        colLabels=HEADERS,
        colWidths=WIDTHS,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(HEAD_BG)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.055)
            continue
        if row == nrow:
            cell.set_facecolor(MEAN_BG)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(mean_colors[col])
        else:
            cell.set_facecolor("#fbfcfc" if row % 2 else "#f2f5f6")
            cell.get_text().set_color(colors[row - 1][col])
        if col == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.03

    fig.suptitle(
        "MiniMax-H3 Video Quantization — VBench-2.0 Results  (100 prompts, 17 dimensions)",
        fontsize=15,
        fontweight="bold",
        y=0.965,
        color="#1a252f",
    )
    ax.set_title(
        "Δ = variant − bf16 baseline   |   N = number of prompts evaluated per dimension   |   "
        "green = improvement over bf16, red = regression",
        fontsize=10,
        color="#566573",
        pad=14,
    )
    fig.subplots_adjust(top=0.895, bottom=0.02, left=0.01, right=0.99)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
