#!/usr/bin/env python3
"""Paired BF16 video NMSE/LPIPS for the rCM-Wan VBench-20 subset."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from eval_rcm_restore_video_similarity import compare


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["nvfp4", "all30"])
    args = parser.parse_args()
    root = args.root.resolve()
    selection = json.loads((root / "selection.json").read_text())
    out = root / "video_similarity"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item_index, item in enumerate(selection):
        prompt_id = f"{item['vbench_index']:04d}"
        reference = root / "generated" / prompt_id / "bf16.mp4"
        for model in args.models:
            candidate = root / "generated" / prompt_id / f"{model}.mp4"
            metrics = compare(reference, candidate, torch.device("cuda"), include_temporal=False)
            row = {"vbench_index": item["vbench_index"], "prompt": item["prompt"],
                   "model": model, "reference": str(reference), "video": str(candidate), **metrics}
            rows.append(row)
            write_csv(out / "per_video.csv", rows)
            print(f"{item_index + 1}/{len(selection)} {prompt_id} {model} "
                  f"nmse={metrics['nmse']:.6g} lpips={metrics['lpips_alex']:.6g}", flush=True)
    summary = []
    for model in args.models:
        group = [row for row in rows if row["model"] == model]
        summary.append({"model": model, "videos": len(group),
                        "mean_nmse": sum(row["nmse"] for row in group) / len(group),
                        "median_nmse": float(torch.tensor([row["nmse"] for row in group]).median()),
                        "mean_lpips_alex": sum(row["lpips_alex"] for row in group) / len(group),
                        "median_lpips_alex": float(torch.tensor([row["lpips_alex"] for row in group]).median())})
    write_csv(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps({"definition": "paired decoded-video metrics against BF16; lower is better", "rows": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
