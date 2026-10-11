#!/usr/bin/env python3
"""Pre-registered E086 timestep analysis for SVDQuant low-rank adaptation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def read_train(path: Path, arm: str):
    data = json.loads(path.read_text())
    rows = []
    for x in data["details"]:
        rows.append({"split": "train", "arm": arm, "prompt_id": int(x["prompt_id"]), "step": int(x["step"]),
                     "timestep": float(x["timestep"]), "video_nmse": float(x["video"]["nmse"]),
                     "audio_nmse": float(x["audio"]["nmse"])})
    return rows


def read_test(root: Path, arm: str):
    rows = []
    for path in sorted((root / "test").glob(f"p*_{arm}.json"), key=lambda x: int(x.name.split("_")[0][1:])):
        data = json.loads(path.read_text())
        for x in data["results"]:
            rows.append({"split": "test", "arm": arm, "prompt_id": int(data["prompt_id"]), "step": int(x["step"]),
                         "timestep": float(x["timestep"]), "video_nmse": float(x["video"]["nmse"]),
                         "audio_nmse": float(x["audio"]["nmse"])})
    return rows


def wilcoxon_greater(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, 0):
        return None
    return float(stats.wilcoxon(values, alternative="greater", zero_method="wilcox", method="auto").pvalue)


def bootstrap_mean(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def holm(items):
    valid = [(i, x["p_value"]) for i, x in enumerate(items) if x["p_value"] is not None]
    for rank, (index, p) in enumerate(sorted(valid, key=lambda x: x[1])):
        items[index]["holm_p_value"] = min(1.0, p * (len(valid) - rank))
    return items


def grouped(rows, arm):
    result = defaultdict(list)
    for row in rows:
        if row["arm"] == arm:
            result[row["prompt_id"]].append(row)
    return {k: sorted(v, key=lambda x: x["step"]) for k, v in result.items()}


def trend(rows):
    curves = grouped(rows, "svdquant_ft")
    slopes = []
    for pid, values in curves.items():
        slopes.append(float(stats.theilslopes(np.log(np.maximum([x["video_nmse"] for x in values], 1e-30)), [x["step"] for x in values]).slope))
    return {"prompt_count": len(slopes), "log_nmse_theil_sen_slope_mean": float(np.mean(slopes)),
            "bootstrap_95_ci": bootstrap_mean(slopes, 861), "one_sided_wilcoxon_p": wilcoxon_greater(-np.asarray(slopes)),
            "interpretation": "negative slope supports lower later-timestep NMSE"}


def benefit(rows):
    plain, ft = grouped(rows, "plain_w4a4"), grouped(rows, "svdquant_ft")
    common = sorted(set(plain) & set(ft))
    differences, late = [], []
    for pid in common:
        p = {x["step"]: x for x in plain[pid]}; f = {x["step"]: x for x in ft[pid]}
        if set(p) != set(f) or 0 not in p:
            raise RuntimeError(f"prompt {pid}: unmatched arms/steps")
        gains = {s: math.log(max(p[s]["video_nmse"], 1e-30)) - math.log(max(f[s]["video_nmse"], 1e-30)) for s in p}
        l = float(np.mean([v for s, v in gains.items() if s != 0]))
        differences.append(gains[0] - l); late.append(l)
    # TOST around ±log(1.10); both one-sided t tests must reject to call it
    # practically equivalent to no more than a ten-percent relative change.
    bound = math.log(1.10)
    n, m, se = len(late), float(np.mean(late)), float(np.std(late, ddof=1) / math.sqrt(len(late))) if len(late) > 1 else float("inf")
    if n > 1 and se > 0 and math.isfinite(se):
        p_lower = float(1 - stats.t.cdf((m + bound) / se, n - 1))
        p_upper = float(stats.t.cdf((m - bound) / se, n - 1))
        equivalent = max(p_lower, p_upper) < .05
    else:
        p_lower = p_upper = None; equivalent = False
    return {"prompt_count": n, "step0_minus_late_log_gain_mean": float(np.mean(differences)),
            "bootstrap_95_ci": bootstrap_mean(differences, 862), "one_sided_wilcoxon_p": wilcoxon_greater(differences),
            "late_mean_log_gain": m, "late_equivalence_bound_log_1p10": bound,
            "late_tost_p_lower": p_lower, "late_tost_p_upper": p_upper, "late_practically_equivalent": equivalent,
            "interpretation": "positive step0-minus-late means fine-tuning improves the first call more"}


def main():
    args = parse_args(); root, out = args.root, args.output
    arms = ("plain_w4a4", "svdquant", "svdquant_ft")
    rows = []
    for arm in arms:
        rows.extend(read_train(root / "train" / f"{arm}.json", arm))
        rows.extend(read_test(root, arm))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "all_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"definition": "teacher-forced complete DiT output NMSE; BF16 has zero error by definition", "splits": {}}
    claims = []
    for split in ("train", "test"):
        subset = [x for x in rows if x["split"] == split]
        result["splits"][split] = {"trend": trend(subset), "first_vs_late_benefit": benefit(subset)}
        claims += [{"name": f"{split}.trend", "p_value": result["splits"][split]["trend"]["one_sided_wilcoxon_p"]},
                   {"name": f"{split}.first_vs_late", "p_value": result["splits"][split]["first_vs_late_benefit"]["one_sided_wilcoxon_p"]}]
    result["holm_family"] = holm(claims)
    result["limitations"] = "Train is in-sample. Held-out uses five independent prompts; confidence intervals and effect sizes are primary because its cluster count is small."
    (out / "analysis.json").write_text(json.dumps(result, indent=2))
    curves = []
    for split in ("train", "test"):
        for arm in arms:
            by_step = defaultdict(list)
            for x in rows:
                if x["split"] == split and x["arm"] == arm:
                    by_step[x["step"]].append(x["video_nmse"])
            for step, values in sorted(by_step.items()):
                curves.append({"split": split, "arm": arm, "step": step, "video_nmse_mean": float(np.mean(values)),
                               "samples": len(values)})
    with (out / "video_nmse_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)

    # The primary comparison is paired within each prompt/call.  Keep the
    # complete per-step gain curve alongside the NMSE curves so the report is
    # auditable without inferring a difference from separately rounded means.
    benefit_rows = []
    for split in ("train", "test"):
        subset = [x for x in rows if x["split"] == split]
        plain = grouped(subset, "plain_w4a4")
        ft = grouped(subset, "svdquant_ft")
        for pid in sorted(set(plain) & set(ft)):
            p, f = {x["step"]: x for x in plain[pid]}, {x["step"]: x for x in ft[pid]}
            if set(p) != set(f):
                raise RuntimeError(f"prompt {pid}: unmatched gain steps")
            for step in sorted(p):
                log_gain = math.log(max(p[step]["video_nmse"], 1e-30)) - math.log(max(f[step]["video_nmse"], 1e-30))
                benefit_rows.append({"split": split, "prompt_id": pid, "step": step,
                                     "log_nmse_gain_ft_vs_plain": log_gain,
                                     "relative_nmse_reduction_ft_vs_plain": 1 - f[step]["video_nmse"] / p[step]["video_nmse"]})
    with (out / "ft_vs_plain_benefit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benefit_rows[0])); writer.writeheader(); writer.writerows(benefit_rows)

    # Small, reviewable manifest: large tensor caches/checkpoints stay on
    # /data1, while this metadata is safe to copy into the repository report.
    contract_path = root / "checkpoint" / "contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    manifest = {
        "train_prompt_ids": sorted({x["prompt_id"] for x in rows if x["split"] == "train"}),
        "test_prompt_ids": sorted({x["prompt_id"] for x in rows if x["split"] == "test"}),
        "train_rows_per_arm": len([x for x in rows if x["split"] == "train" and x["arm"] == "plain_w4a4"]),
        "test_rows_per_arm": len([x for x in rows if x["split"] == "test" and x["arm"] == "plain_w4a4"]),
        "arms": list(arms),
        "fine_tune_contract": contract,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
