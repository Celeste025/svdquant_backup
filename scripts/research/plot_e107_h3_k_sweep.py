#!/usr/bin/env python3
"""Plot exact pooled NMSE as a function of prior true-error subtraction gain k."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/research/E107"
SOURCE = BASE / "previous_error_subtraction.json"
COLORS = {"race": "#2955A4", "clap0": "#C66A27"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = json.loads(SOURCE.read_text())
    if data["format"] != "h3-e107-previous-step-oracle-error-subtraction-v1":
        raise RuntimeError("unexpected E107 source schema")
    figure_path = BASE / "figures/k_sweep_pooled_nmse.png"
    json_path = BASE / "k_sweep.json"
    csv_path = BASE / "k_sweep.csv"
    if any(path.exists() for path in (figure_path, json_path, csv_path)):
        raise FileExistsError("refusing to overwrite E107 k-sweep outputs")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1.5, 151, dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.1), sharex=True)
    figure.patch.set_facecolor("white")
    sweep = {"format": "h3-e107-pooled-k-sweep-v1",
             "source": str(SOURCE.resolve()), "source_sha256": sha256(SOURCE),
             "definition": "pooled NMSE over saved BF16-trajectory paired steps 1..19; sum ||e_t-k e_(t-1)||^2 / sum ||BF16_t||^2",
             "k_grid": grid.tolist(), "cases": {}}
    csv_rows = []
    for column, case in enumerate(("race", "clap0")):
        sweep["cases"][case] = {}
        for row_index, modality in enumerate(("video", "audio")):
            rows = [row for row in data["cases"][case]["per_step"] if row["modality"] == modality]
            if len(rows) != 19:
                raise RuntimeError(f"incomplete E107 series: {case}/{modality}")
            numerator_0 = sum(row["error_energy"] for row in rows)
            cross = sum(row["cross_energy"] for row in rows)
            prior = sum(row["previous_error_energy"] for row in rows)
            denominator = sum(row["reference_energy"] for row in rows)
            nmse = (numerator_0 - 2 * grid * cross + grid**2 * prior) / denominator
            if np.any(nmse <= 0) or not np.isfinite(nmse).all():
                raise RuntimeError("invalid k-sweep NMSE")
            k_opt = cross / prior
            nmse_opt = (numerator_0 - cross**2 / prior) / denominator
            baseline = numerator_0 / denominator
            expected = data["cases"][case]["summary"][modality]
            if (abs(baseline - expected["baseline_pooled_nmse"]) > 1e-12 or
                abs(nmse[50] - expected["k05_pooled_nmse"]) > 1e-12 or
                abs(nmse[80] - expected["k08_pooled_nmse"]) > 1e-12):
                raise RuntimeError("curve did not reproduce independently stored k=0/0.5/0.8 values")
            sweep["cases"][case][modality] = {
                "baseline_pooled_nmse": baseline, "pooled_optimal_k_oracle": k_opt,
                "pooled_optimal_nmse": nmse_opt,
                "pooled_optimal_relative_reduction": 1 - nmse_opt / baseline,
                "nmse_grid": nmse.tolist(),
            }
            for k, value in zip(grid, nmse, strict=True):
                csv_rows.append({"case": case, "modality": modality, "k": float(k),
                                 "pooled_nmse": float(value),
                                 "relative_reduction": float(1 - value / baseline)})
            ax = axes[row_index, column]
            ax.set_facecolor("#FBFCFD")
            ax.plot(grid, nmse * 100, color=COLORS[case], linewidth=2.5, label="Pooled NMSE")
            ax.axhline(baseline * 100, color="#747B85", linestyle="--", linewidth=1.1)
            ax.scatter([0.5, 0.8], [nmse[50] * 100, nmse[80] * 100],
                       color=COLORS[case], marker="o", s=49, zorder=4)
            ax.scatter([k_opt], [nmse_opt * 100], facecolor="white", edgecolor="#20262E",
                       linewidth=1.6, marker="D", s=62, zorder=5)
            ax.annotate(f"best k={k_opt:.2f}", (k_opt, nmse_opt * 100),
                        xytext=(0, 11), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color="#20262E")
            ax.set_title(f"{'Race' if case == 'race' else 'Clap'} · {modality}",
                         loc="left", fontsize=13, fontweight="bold", color="#20262E")
            ax.grid(axis="y", color="#DDE2E8", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xlim(0, 1.5)
            ax.set_ylabel("Pooled output NMSE (%)", fontsize=10)
            ax.tick_params(labelsize=9)
            if row_index == 1:
                ax.set_xlabel("k applied to previous true output error", fontsize=10)
    figure.suptitle("Previous-step oracle error subtraction · 19 paired steps per case",
                     x=0.08, y=0.985, ha="left", fontsize=15, fontweight="bold", color="#20262E")
    figure.text(0.08, 0.022,
                "Dashed: no correction. Circles: k=0.5 and 0.8. Diamond: pooled least-squares optimum."
                " Separate y-scales; true prior BF16 error is unavailable in normal quantized inference.",
                fontsize=9.2, color="#505B66")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.11, wspace=0.23, hspace=0.26)
    figure.savefig(figure_path, dpi=175, facecolor="white")
    plt.close(figure)
    with json_path.open("x") as stream:
        json.dump(sweep, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "modality", "k", "pooled_nmse", "relative_reduction"))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({case: {modality: {key: value for key, value in item.items() if key != "nmse_grid"}
                             for modality, item in modalities.items()}
                      for case, modalities in sweep["cases"].items()}, indent=2))
    print(f"COMPLETE {figure_path} {json_path} {csv_path}")


if __name__ == "__main__":
    main()
