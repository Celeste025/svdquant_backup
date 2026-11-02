#!/usr/bin/env python3
"""Aggregate audited E129 video distances, keeping E124/E126 suites separate."""
from __future__ import annotations

import csv
import json
import math

from finalize_e129_h3_svdquant_attention import E129


def main() -> None:
    audit = json.loads((E129 / "final_audit.json").read_text())
    if audit["complete_videos"] != audit["expected_videos"] or audit["expected_videos"] != 3:
        raise RuntimeError("all three selected videos must pass audit before aggregation")
    rows = []
    for case in sorted(audit["cases"]):
        path = E129 / case / "video_metrics.json"
        metric = json.loads(path.read_text())
        if metric["case"] != case or metric["frames"] != 124:
            raise RuntimeError(f"invalid video metrics: {case}")
        summary = metric["summary"]
        plain = summary["plain_w4a4_vs_bf16"]
        svd = summary["svdquant_vs_bf16"]
        direct = summary["svdquant_vs_plain_w4a4"]
        row = {"suite": "12-prompt" if case.startswith("p") else "VBench5",
               "case": case, "plain_rgb_nmse": plain["rgb_nmse"],
               "svd_rgb_nmse": svd["rgb_nmse"],
               "svd_vs_plain_rgb_nmse": direct["rgb_nmse"],
               "svd_rgb_relative_change": svd["rgb_nmse"] / plain["rgb_nmse"] - 1,
               "plain_lpips": plain["lpips_alex"],
               "svd_lpips": svd["lpips_alex"],
               "svd_vs_plain_lpips": direct["lpips_alex"],
               "svd_lpips_relative_change": svd["lpips_alex"] / plain["lpips_alex"] - 1,
               "svd_sum_dit_seconds": audit["cases"][case]["sum_dit_seconds"]}
        if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise RuntimeError(f"nonfinite metric: {case}")
        rows.append(row)
    csv_path = E129 / "video_metrics_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for suite in ("12-prompt", "VBench5", "all"):
        selected = [row for row in rows if suite == "all" or row["suite"] == suite]
        aggregate[suite] = {"n": len(selected),
                            **{key: sum(row[key] for row in selected) / len(selected)
                               for key in rows[0] if key not in ("suite", "case")}}
    result = {"format": "e129-svdquant-video-summary-v1", "cases": rows,
              "aggregate": aggregate,
              "caution": "Frame-paired free-rollout RGB NMSE and LPIPS measure distance to a separately generated BF16 video, not teacher-forced DiT quantization error or human video quality. SVDQuant uses BF16-GEMM fake NVFP4; E128 plain W4A4 uses native FP4 GEMM, so timings are not comparable."}
    target = E129 / "video_metrics_summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"csv": str(csv_path), "json": str(target),
                      "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
