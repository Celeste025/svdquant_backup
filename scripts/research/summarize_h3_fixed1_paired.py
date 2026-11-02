#!/usr/bin/env python3
"""Validate and summarize the paired E100 dynamic-vs-fixed scale result."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def component_summary(dynamic: list[dict], fixed: list[dict], name: str) -> dict:
    d = [row[name] for row in dynamic]
    f = [row[name] for row in fixed]
    d_nmse = [row["nmse"] for row in d]
    f_nmse = [row["nmse"] for row in f]
    d_mean = statistics.mean(d_nmse)
    f_mean = statistics.mean(f_nmse)
    d_pooled = sum(row["mse"] for row in d) / sum(row["mse"] / row["nmse"] for row in d)
    f_pooled = sum(row["mse"] for row in f) / sum(row["mse"] / row["nmse"] for row in f)
    per_step = [{"step": i, "dynamic_nmse": a, "fixed_nmse": b,
                 "fixed_over_dynamic": b / a, "fixed_minus_dynamic": b - a}
                for i, (a, b) in enumerate(zip(d_nmse, f_nmse, strict=True))]
    phases = {}
    for phase, start, end in (("early_0_4", 0, 5), ("middle_5_14", 5, 15), ("late_15_19", 15, 20)):
        a = statistics.mean(d_nmse[start:end])
        b = statistics.mean(f_nmse[start:end])
        phases[phase] = {"dynamic_mean_nmse": a, "fixed_mean_nmse": b,
                         "fixed_over_dynamic": b / a}
    return {"dynamic_mean_nmse": d_mean, "fixed_mean_nmse": f_mean,
            "mean_nmse_fixed_over_dynamic": f_mean / d_mean,
            "dynamic_pooled_nmse": d_pooled, "fixed_pooled_nmse": f_pooled,
            "pooled_nmse_fixed_over_dynamic": f_pooled / d_pooled,
            "fixed_lower_steps": sum(b < a for a, b in zip(d_nmse, f_nmse, strict=True)),
            "fixed_higher_steps": sum(b > a for a, b in zip(d_nmse, f_nmse, strict=True)),
            "fixed_equal_steps": sum(b == a for a, b in zip(d_nmse, f_nmse, strict=True)),
            "phases": phases, "per_step": per_step}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--video-pixels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dynamic = load(args.dynamic)
    fixed = load(args.fixed)
    for key in ("prompt_id", "seed", "height", "width", "frames", "steps"):
        if dynamic[key] != fixed[key]:
            raise RuntimeError(f"unpaired metadata: {key}")
    if (dynamic["prompt_id"], dynamic["seed"], dynamic["height"], dynamic["width"],
        dynamic["frames"], dynamic["steps"]) != (1, 22529, 480, 832, 124, 20):
        raise RuntimeError("unexpected E100 case")
    d_rows, f_rows = dynamic["results"], fixed["results"]
    if len(d_rows) != 20 or len(f_rows) != 20:
        raise RuntimeError("incomplete step coverage")
    for i, (a, b) in enumerate(zip(d_rows, f_rows, strict=True)):
        if a["step"] != i or b["step"] != i or a["timestep"] != b["timestep"]:
            raise RuntimeError(f"unpaired step {i}")
        for row in (a, b):
            for component in ("video", "audio"):
                for key in ("nmse", "mse", "rmse", "max_abs", "cosine"):
                    if not math.isfinite(row[component][key]):
                        raise RuntimeError(f"nonfinite {component}.{key} at step {i}")
                if row[component]["nmse"] <= 0:
                    raise RuntimeError(f"invalid {component} NMSE at step {i}")
    pixels = load(args.video_pixels)
    if pixels["frames"] != 124:
        raise RuntimeError("incomplete decoded video comparison")

    summary = {"case": {key: dynamic[key] for key in ("prompt_id", "seed", "height", "width", "frames", "steps")},
               "definition": "teacher-forced full-DiT output error against shared BF16 calls; lower NMSE is better",
               "video": component_summary(d_rows, f_rows, "video"),
               "audio": component_summary(d_rows, f_rows, "audio"),
               "decoded_video": pixels["summary"],
               "sources": {"dynamic": str(args.dynamic), "fixed": str(args.fixed),
                           "video_pixels": str(args.video_pixels)}}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ("video", "audio"):
        s = summary[name]
        print(f"{name}: mean NMSE {s['dynamic_mean_nmse']:.6g} -> {s['fixed_mean_nmse']:.6g} "
              f"({(s['mean_nmse_fixed_over_dynamic'] - 1) * 100:+.3f}%); "
              f"pooled {(s['pooled_nmse_fixed_over_dynamic'] - 1) * 100:+.3f}%; "
              f"fixed lower/higher steps {s['fixed_lower_steps']}/{s['fixed_higher_steps']}")


if __name__ == "__main__":
    main()
