#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = Path("/home/admin/workspace/aop_lab/app_data/runs")
OUTPUT_DIR = Path("/home/admin/workspace/aop_lab/app_data/visionreward_handoff")
WEIGHT_PATH = Path("/tmp/VisionReward-repo/VisionReward_Video/weight.json")

RCM_LABELS = {
    "bf16": "BF16 (reference)",
    "int4_plain": "INT4 W4A4 (plain)",
    "int4_svdquant": "INT4 + SVDQuant",
    "nvfp4": "NVFP4 W4A4 (plain)",
    "nvfp4_svdquant": "NVFP4 + SVDQuant",
    "nvfp4_svdquant_g10_r64": "NVFP4 + SVDQuant (g10 / r64)",
    "nvfp4_svdquant_g20_r32": "NVFP4 + SVDQuant (g20 / r32)",
}
H3_LABELS = {
    "bf16": "BF16 (reference)",
    "w4a4": "W4A4 (plain)",
    "svdquant_r32": "SVDQuant rank-32",
    "svdquant_r64": "SVDQuant rank-64",
    "convrot": "ConvRot",
}

TOPICS = [
    ("Prompt\nAlignment", [0, 1, 2]),
    ("Comp &\nCamera", [3, 4, 5]),
    ("Color", [6]),
    ("Lighting", list(range(7, 13))),
    ("Shape\nConsistency", list(range(13, 18))),
    ("Camera\nMotion", [18, 19]),
    ("Motion\nSmoothness", [20]),
    ("Motion\nRealism", [21]),
    ("Stability", [22]),
    ("Detail", [23, 24, 25]),
    ("Text", [26, 27]),
    ("Physical\nWorld", [28]),
]


def load_binary_answers():
    records = []
    for path in [RUNS / "visionreward-vbench51/scores_all.jsonl", RUNS / "visionreward-convrot/scores_convrot.jsonl"]:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    scores = {}
    for model, labels in [("rcm_wan", RCM_LABELS), ("minimax_h3", H3_LABELS)]:
        for variant in labels:
            rows = [r for r in records if r["model"] == model and r["variant"] == variant]
            if len(rows) != 51:
                raise ValueError(f"expected 51 rows for {model}/{variant}, got {len(rows)}")
            scores.setdefault(model, {})[variant] = [r["answers"] for r in rows]
    return scores


def topic_scores(answer_rows, weights):
    values = []
    for _, question_ids in TOPICS:
        total = 0.0
        for answers in answer_rows:
            for qid in question_ids:
                total += weights[qid] * answers[qid]
        values.append(total / (len(answer_rows) * sum(weights[qid] for qid in question_ids)))
    overall = sum(sum(weights[i] * a[i] for i in range(29)) for a in answer_rows) / (len(answer_rows) * len(weights))
    return values, overall


def render_table(model, labels, scores, weights, title, output_path):
    evaluated = {v: topic_scores(scores[model][v], weights) for v in labels}
    ordered = ["bf16"] + sorted((v for v in labels if v != "bf16"), key=lambda v: -evaluated[v][1])

    n_topics = len(TOPICS)
    cell_text = [[f"{v:+.4f}" for v in evaluated[variant][0]] + [f"{evaluated[variant][1]:+.4f}"] for variant in ordered]
    row_labels = [labels[variant] for variant in ordered]

    fig, ax = plt.subplots(figsize=(16.4, 0.6 * (len(ordered) + 1) + 1.7))
    ax.axis("off")

    header = [name for name, _ in TOPICS] + ["Overall"]
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=header,
                     colWidths=[0.063] * n_topics + [0.078], cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.5)

    for column in range(n_topics + 1):
        cell = table[0, column]
        cell.set_facecolor("#1f4e79")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        cell.set_height(cell.get_height() * 1.7)
    for row, variant in enumerate(ordered, start=1):
        facecolor = "#fff2cc" if variant == "bf16" else ("#f2f6fb" if row % 2 == 0 else "white")
        for column in range(n_topics + 1):
            table[row, column].set_facecolor(facecolor)
        row_label = table[row, -1]
        row_label.set_text_props(ha="left", fontsize=10)
        row_label.set_width(row_label.get_width() * 2.3)
        table[row, n_topics].set_text_props(fontweight="bold", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    fig.text(0.5, 0.012,
             "Official 29-question checklist, binary yes/no answers; each topic score is weight-normalized within its questions; mean over 51 VBench cases",
             ha="center", fontsize=8.2, color="#555555")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    scores = load_binary_answers()
    weights = json.loads(WEIGHT_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_table("rcm_wan", RCM_LABELS, scores, weights,
                 "rCM-Wan on VBench-51 — VisionReward by Topic (binary, 29Q)",
                 OUTPUT_DIR / "rcm_wan_visionreward_binary.png")
    render_table("minimax_h3", H3_LABELS, scores, weights,
                 "MiniMax-H3 on VBench-51 — VisionReward by Topic (binary, 29Q)",
                 OUTPUT_DIR / "h3_visionreward_binary.png")
    print(f"written: {OUTPUT_DIR / 'rcm_wan_visionreward_binary.png'}")
    print(f"written: {OUTPUT_DIR / 'h3_visionreward_binary.png'}")


if __name__ == "__main__":
    main()
