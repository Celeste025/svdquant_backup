#!/usr/bin/env python3
"""Compute decoded-frame MSE/NMSE/PSNR for the H3 BF16-restore sweep."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "results" / "samples"
OUT = ROOT / "results" / "reports" / "minimax_h3_svdquant_restore_top"


def one(reference: Path, candidate: Path) -> dict[str, float | int]:
    ref_cap, cand_cap = cv2.VideoCapture(str(reference)), cv2.VideoCapture(str(candidate))
    if not ref_cap.isOpened() or not cand_cap.isOpened():
        raise RuntimeError(f"could not decode {reference} or {candidate}")
    err2 = power = 0.0
    count = frames = 0
    while True:
        aok, a = ref_cap.read(); bok, b = cand_cap.read()
        if aok != bok:
            raise RuntimeError(f"frame mismatch: {reference} vs {candidate}")
        if not aok:
            break
        if a.shape != b.shape:
            raise RuntimeError(f"shape mismatch: {a.shape} vs {b.shape}")
        a = a.astype(np.float64) / 255.0; b = b.astype(np.float64) / 255.0
        err2 += float(np.square(b - a).sum()); power += float(np.square(a).sum())
        count += a.size; frames += 1
    ref_cap.release(); cand_cap.release()
    mse = err2 / count
    return {"frames": frames, "mse": mse, "nmse": err2 / power,
            "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-30))}


def main() -> None:
    rows: list[dict[str, object]] = []
    for pid, seed in ((2, 30209), (16, 20026), (26, 63583)):
        base = SAMPLES / "minimax_h3_svdquant_standard_8p64s" / f"p{pid}"
        ref = base / f"bf16_id{pid}_seed{seed}_576x1024_124f_20steps.mp4"
        variants = {
            "w4a4": base / f"w4a4_id{pid}_seed{seed}_576x1024_124f_20steps.mp4",
            "svdquant": base / f"svdquant_id{pid}_seed{seed}_576x1024_124f_20steps.mp4",
            "restore_top10": SAMPLES / "minimax_h3_svdquant_restore_top" / f"p{pid}_top10" / f"restore_top10pct_id{pid}_seed{seed}_576x1024_124f_20steps.mp4",
            "restore_top20": SAMPLES / "minimax_h3_svdquant_restore_top" / f"p{pid}_top20" / f"restore_top20pct_id{pid}_seed{seed}_576x1024_124f_20steps.mp4",
        }
        for name, path in variants.items():
            row: dict[str, object] = {"prompt_id": pid, "variant": name, "reference": str(ref), "video": str(path)}
            row.update(one(ref, path)); rows.append(row)
            print(pid, name, f"NMSE={100 * row['nmse']:.3f}%", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
