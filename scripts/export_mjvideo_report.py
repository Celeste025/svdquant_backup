#!/usr/bin/env python3
"""Export review-friendly MJ-VIDEO tables while omitting safety and bias aspects."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ["case_id", "prompt", "variant", "mjvideo_total", "alignment", "fineness", "coherence_consistency"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, help="When set, also write mjvideo_metrics.json into each case directory.")
    parser.add_argument("--case-dir-template", default="cases/{case_id}",
                        help="Path relative to --case-root for each case's review JSON.")
    args = parser.parse_args()

    doc = json.loads(args.results.read_text())
    rows = []
    for case_id, case in sorted(doc["results"].items()):
        for variant, result in case["variants"].items():
            aspects = result["aspects"]
            rows.append({
                "case_id": case_id,
                "prompt": case["prompt"],
                "variant": variant,
                "mjvideo_total": result["score"],
                "alignment": aspects["alignment"],
                "fineness": aspects["fineness"],
                "coherence_consistency": aspects["coherence_consistency"],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = args.output_dir / "mjvideo_prompt_metrics.csv"
    with detail.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = args.output_dir / "mjvideo_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "case_count", "mjvideo_total", "alignment", "fineness", "coherence_consistency"])
        writer.writeheader()
        for variant, values in sorted(grouped.items()):
            writer.writerow({
                "variant": variant,
                "case_count": len(values),
                **{key: sum(float(row[key]) for row in values) / len(values) for key in FIELDS[3:]},
            })

    if args.case_root:
        for case_id, case in doc["results"].items():
            output = {
                "prompt": case["prompt"],
                "metrics": {
                    variant: {
                        "mjvideo_total": result["score"],
                        "alignment": result["aspects"]["alignment"],
                        "fineness": result["aspects"]["fineness"],
                        "coherence_consistency": result["aspects"]["coherence_consistency"],
                    }
                    for variant, result in case["variants"].items()
                },
                "excluded_metrics": ["safety", "bias_fairness"],
            }
            case_dir = args.case_root / args.case_dir_template.format(case_id=case_id)
            if not case_dir.is_dir():
                raise FileNotFoundError(case_dir)
            (case_dir / "mjvideo_metrics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
