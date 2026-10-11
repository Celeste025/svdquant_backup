#!/usr/bin/env python3
"""Render lightweight SVG figures from the audited E086 CSV artifacts."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


COLORS = {"plain_w4a4": "#7a7a7a", "svdquant": "#e07a2d", "svdquant_ft": "#277da1"}
LABELS = {"plain_w4a4": "plain W4A4", "svdquant": "SVDQuant", "svdquant_ft": "SVDQuant + FT"}


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_csv(path: Path):
    with path.open(newline="") as h:
        return list(csv.DictReader(h))


def ticks(lo: float, hi: float, count: int = 5):
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def figure_shell(width: int, height: int, title: str, body: str):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">E086 MiniMax-H3 timestep comparison.</desc>
<rect width="100%" height="100%" fill="white"/><style>
text{{font-family:Arial,sans-serif;fill:#202124}} .small{{font-size:12px}} .label{{font-size:14px}} .title{{font-size:18px;font-weight:600}} .grid{{stroke:#d9dce1;stroke-width:1}} .axis{{stroke:#60656f;stroke-width:1.2}} .dash{{stroke-dasharray:5 4}}
</style>{body}</svg>'''


def panel_lines(x: int, y: int, w: int, h: int, title: str, curves: dict[str, list[tuple[int, float]]], y_label: str, y_min: float, y_max: float, percent=False):
    vals = [p[0] for points in curves.values() for p in points]
    x_min, x_max = min(vals), max(vals)
    sx = lambda v: x + 54 + (v - x_min) / max(x_max - x_min, 1) * (w - 72)
    sy = lambda v: y + 26 + (y_max - v) / max(y_max - y_min, 1e-12) * (h - 64)
    out = [f'<text class="label" x="{x+54}" y="{y+17}">{escape(title)}</text>']
    for value in ticks(y_min, y_max):
        yy = sy(value); label = f"{value * 100:.0f}%" if percent else f"{value:.02f}"
        out.append(f'<line class="grid" x1="{x+54}" y1="{yy:.1f}" x2="{x+w-18}" y2="{yy:.1f}"/><text class="small" text-anchor="end" x="{x+48}" y="{yy+4:.1f}">{label}</text>')
    if y_min < 0 < y_max:
        yy = sy(0); out.append(f'<line class="axis dash" x1="{x+54}" y1="{yy:.1f}" x2="{x+w-18}" y2="{yy:.1f}"/>')
    out += [f'<line class="axis" x1="{x+54}" y1="{y+26}" x2="{x+54}" y2="{y+h-38}"/>', f'<line class="axis" x1="{x+54}" y1="{y+h-38}" x2="{x+w-18}" y2="{y+h-38}"/>']
    for step in sorted(set(vals)):
        xx = sx(step); out.append(f'<text class="small" text-anchor="middle" x="{xx:.1f}" y="{y+h-21}">{step}</text>')
    out.append(f'<text class="small" text-anchor="middle" x="{x+w/2:.1f}" y="{y+h-4}">denoising step</text>')
    out.append(f'<text class="small" text-anchor="middle" transform="translate({x+13},{y+h/2:.1f}) rotate(-90)">{escape(y_label)}</text>')
    for arm, points in curves.items():
        path = " ".join(("M" if i == 0 else "L") + f"{sx(step):.1f},{sy(value):.1f}" for i, (step, value) in enumerate(points))
        out.append(f'<path d="{path}" fill="none" stroke="{COLORS[arm]}" stroke-width="2.5"/>')
        out.extend(f'<circle cx="{sx(step):.1f}" cy="{sy(value):.1f}" r="3" fill="{COLORS[arm]}"/>' for step, value in points)
    return "".join(out)


def main():
    args = args_parser(); args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = read_csv(args.analysis_dir / "video_nmse_curves.csv")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in curves:
        grouped[r["split"]][r["arm"]].append((int(r["step"]), float(r["video_nmse_mean"])))
    for split in grouped:
        for arm in grouped[split]: grouped[split][arm].sort()
    legend = "".join(f'<line x1="{80+i*180}" y1="44" x2="{104+i*180}" y2="44" stroke="{COLORS[a]}" stroke-width="3"/><text class="small" x="{110+i*180}" y="48">{LABELS[a]}</text>' for i,a in enumerate(("plain_w4a4", "svdquant", "svdquant_ft")))
    body = '<text class="title" x="40" y="27">MiniMax-H3 E086 · video NMSE by denoising step</text>' + legend
    body += panel_lines(34, 62, 570, 350, "Calibration / fine-tuning calls (in-sample; n=8 prompts)", grouped["train"], "mean video NMSE", 0, .21)
    body += panel_lines(620, 62, 570, 350, "Held-out prompts (n=5)", grouped["test"], "mean video NMSE", 0, .21)
    (args.output_dir / "video_nmse_by_step.svg").write_text(figure_shell(1220, 435, "Video NMSE by step", body))

    benefit = read_csv(args.analysis_dir / "ft_vs_plain_benefit.csv")
    gains = defaultdict(lambda: defaultdict(list))
    for r in benefit:
        gains[r["split"]][int(r["step"])].append(float(r["relative_nmse_reduction_ft_vs_plain"]))
    gain_curves = {split: {"svdquant_ft": sorted((step, sum(v) / len(v)) for step, v in by_step.items())} for split, by_step in gains.items()}
    body = '<text class="title" x="40" y="27">SVDQuant + FT improvement over plain W4A4</text><text class="small" x="40" y="48">positive = lower video NMSE after fine-tuning; line is prompt-average paired relative reduction</text>'
    body += panel_lines(34, 62, 570, 350, "Calibration / fine-tuning calls (in-sample)", gain_curves["train"], "NMSE reduction", -.10, .25, percent=True)
    body += panel_lines(620, 62, 570, 350, "Held-out prompts", gain_curves["test"], "NMSE reduction", -.10, .25, percent=True)
    (args.output_dir / "svdquant_ft_gain_by_step.svg").write_text(figure_shell(1220, 435, "SVDQuant FT improvement by step", body))


if __name__ == "__main__":
    main()
