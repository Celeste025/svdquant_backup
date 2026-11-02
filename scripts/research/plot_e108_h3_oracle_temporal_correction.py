#!/usr/bin/env python3
"""Plot verified E108 closed-loop DiT output NMSE for both cases."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    reports = ROOT / "results/research/E108"
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True)
    for row, case in enumerate(("race", "clap0")):
        path = reports / case / "oracle_prev_error_k06_denoise.json"
        report = json.loads(path.read_text())
        if (report.get("format"), report.get("status"), report.get("case"),
            len(report.get("steps", []))) != (
            "h3-e108-oracle-previous-error-denoise-v1", "denoise_complete", case, 20
        ):
            raise RuntimeError(f"incomplete E108 report: {case}")
        for col, modality in enumerate(("video", "audio")):
            raw = np.array([step["modalities"][modality]["target_nmse"] for step in report["steps"]]) * 100
            corrected = np.array([step["modalities"][modality]["realized_nmse"] for step in report["steps"]]) * 100
            ax = axes[row, col]
            ax.plot(range(20), raw, "o-", color="#c95537", linewidth=1.8, markersize=3.3,
                    label="Raw SVDQuant on E108 inputs")
            ax.plot(range(20), corrected, "o-", color="#19789a", linewidth=1.8, markersize=3.3,
                    label="After oracle correction, k=0.6")
            ax.set_title(f"{'Race' if row == 0 else 'Clap'} · {modality.capitalize()}")
            ax.set_ylabel("Paired complete-DiT output NMSE (%)")
            ax.set_xlim(0, 19)
            ax.set_xticks((0, 3, 6, 9, 12, 15, 19))
            ax.grid(alpha=0.22)
            if row == 1:
                ax.set_xlabel("Denoising step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99), frameon=False)
    figure.suptitle("H3 closed-loop oracle previous-error correction", fontsize=15, y=1.035)
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2)
    dest = reports / "oracle_prev_error_k06_step_nmse.png"
    if dest.exists():
        raise FileExistsError(dest)
    figure.savefig(dest, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(dest)


if __name__ == "__main__":
    main()
