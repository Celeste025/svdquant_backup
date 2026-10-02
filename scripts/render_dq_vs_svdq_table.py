#!/usr/bin/env python3
"""Render the SVDQuant vs DeltaQuant single-layer summary table to PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("/home/admin/workspace/aop_lab/app_data")
OUT = DATA_ROOT / "figures" / "h3_dq_vs_svdq_singlelayer_table.png"
DARK, NOTE = "#2C3E50", "#566573"

rows = [
    ("SVDQuant",          "0.4664%", "0.8961%"),
    ("DeltaQuant cube16", "0.4522%", "0.8909%"),
    ("DeltaQuant cube32", "0.4778%", "0.9077%"),
    ("DeltaQuant cube64", "0.4991%", "0.9222%"),
]
best = {1: 1, 2: 1}  # metric column -> row index of the lowest (best) value

fig, ax = plt.subplots(figsize=(7.4, 2.6), dpi=200)
ax.axis("off")

headers = ["Method", "act NMSE (video) ↓", "layer-output NMSE ↓"]
table = ax.table(
    cellText=[list(r) for r in rows],
    colLabels=headers,
    cellLoc="center",
    colLoc="center",
    loc="center",
    colWidths=[0.36, 0.32, 0.32],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

for j in range(len(headers)):
    cell = table[0, j]
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_facecolor(DARK)
    cell.set_edgecolor("white")
    cell.set_linewidth(1.2)

for i in range(len(rows)):
    for j in range(3):
        cell = table[i + 1, j]
        cell.set_edgecolor("#C9D1DC")
        cell.set_linewidth(0.8)
        if i % 2 == 1:
            cell.set_facecolor("#F2F5FA")
        if i == 0:
            cell.get_text().set_fontweight("bold")
        if j in best and best[j] == i:
            cell.get_text().set_color("#B30000")
            cell.get_text().set_fontweight("bold")

fig.suptitle("SVDQuant vs DeltaQuant — Single-Layer Quantization Error (blocks.24.mlp.fc2)",
             fontsize=13, fontweight="bold", y=0.98, color=DARK)
ax.set_title("Mean over 3 prompts × 8 denoising steps · same NVFP4 weights, only activation quantization differs",
             fontsize=9, color=NOTE, pad=12)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
