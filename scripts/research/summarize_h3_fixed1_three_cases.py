#!/usr/bin/env python3
"""Summarize E101's validated three-case teacher-forced and video results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E101"
CASES = {"race": (16, 20026), "clap0": (161, 38001610), "clap1": (161, 38001611)}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    output = REPORT / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case, (prompt_id, seed) in CASES.items():
        dynamic = read(REPORT / case / "svdquant.json")
        fixed = read(REPORT / case / "svdquant_fixed1.json")
        video = read(REPORT / case / "video_pixel_comparison.json")
        if any(report["prompt_id"] != prompt_id or report["seed"] != seed or
               report["steps"] != 20 or len(report["results"]) != 20 for report in (dynamic, fixed)):
            raise ValueError(f"Teacher-forced identity/completeness mismatch for {case}")
        if video["seed"] != seed or video["frames"] != 124:
            raise ValueError(f"Decoded-video identity/completeness mismatch for {case}")
        if [step["timestep"] for step in dynamic["results"]] != [step["timestep"] for step in fixed["results"]]:
            raise ValueError(f"Timestep mismatch for {case}")
        result = {"case": case, "prompt_id": prompt_id, "seed": seed, "teacher_forced": {},
                  "decoded_rgb": video["summary"], "montage": video["montage"]}
        for modality in ("video", "audio"):
            mean = {}
            for arm, report in (("dynamic", dynamic), ("fixed", fixed)):
                values = [step[modality]["nmse"] for step in report["results"]]
                if any(not 0 <= value < 10 for value in values):
                    raise ValueError(f"Nonfinite or implausible {modality} NMSE for {case}/{arm}")
                mean[arm] = sum(values) / len(values)
            mean["fixed_over_dynamic"] = mean["fixed"] / mean["dynamic"]
            mean["fixed_minus_dynamic_pct_of_dynamic"] = 100 * (mean["fixed_over_dynamic"] - 1)
            mean["steps_fixed_lower"] = sum(
                f[modality]["nmse"] < d[modality]["nmse"]
                for d, f in zip(dynamic["results"], fixed["results"], strict=True))
            result["teacher_forced"][modality] = mean
        rows.append(result)
    payload = {"definition": "Teacher-forced means same BF16 trajectory DiT inputs; decoded RGB is free-rollout H.264 distance and must not be read as DiT NMSE.",
               "cases": rows}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
