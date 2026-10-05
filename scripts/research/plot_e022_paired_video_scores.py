"""Plot the already verified E022 readout; CPU only, no new model evaluation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/research/E022/video_summary.json"
DEST = Path("/data1/models/svdquant-wjq/research/20261003/E022/learning_curves")


def main():
    data = json.loads(SOURCE.read_text())
    assert data["status"] == "complete" and data["phase"] == "complete"
    names = ["128 Shuttle", "134 Iron Man", "095 Mars sunrise", "232 Museum",
             "133 Reading", "191 Courtyard", "204 Harbor", "197 Football field"]
    order = data["protocol"]["prompt_order"]
    fig, ax = plt.subplots(figsize=(9, 5.7), constrained_layout=True)
    for y, pid in enumerate(order):
        p = data["per_prompt"][pid]
        values = []
        for tid in p["trajectory_ids"]:
            a = data["per_trajectory"][tid]["arms"]
            values.append(a["qad_native_dev_selected"]["metrics"]["mj_total"]
                          - a["plain_step0000"]["metrics"]["mj_total"])
        mean = p["paired_differences"]["qad_minus_plain"]["mj_total"]
        ax.plot(values, [y, y], color="#acb8c5", linewidth=2, zorder=1)
        ax.scatter(values, [y, y], s=32, color="#748699", zorder=2,
                   label="Individual seeds" if y == 0 else None)
        ax.scatter([mean], [y], marker="D", s=48, color="#126c8c", zorder=3,
                   label="Two-seed mean" if y == 0 else None)
    overall = data["paired_mean_differences"]["qad_minus_plain"]["mj_total"]
    ax.axvline(0, color="#303b44", linewidth=1)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("MJ total score difference: selected QAD64 minus plain NVFP4")
    ax.set_title(f"E022: opposing changes yield an overall mean of {overall:+.6f}")
    ax.grid(axis="x", color="#e5e9ee")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.supxlabel("8 fixed prompts, 2 seeds each. All failures retained; dots are observations, not confidence intervals.",
                  fontsize=9)
    DEST.mkdir(parents=True, exist_ok=True)
    output = DEST / "video_paired_mj.png"
    fig.savefig(output, dpi=180)
    receipt = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "plot": str(output), "plot_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "policy": "Plot verified summary only; no filtering or added evaluation."
    }
    (ROOT / "results/research/E022/video_paired_mj_plot.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
