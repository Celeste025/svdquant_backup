#!/usr/bin/env python3
"""Extract same-scope Sage2++/Sage3 counters for 22,677-token H3 attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_e098_sage_ncu import read_ncu
from summarize_e130_ncu import METRICS

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/data1/models/svdquant-wjq/research/20261025/E130")
REPORTS = {"sage2pp": RAW / "sage2pp_long_core_gpu2.ncu-rep",
           "sage3_maskfix": RAW / "sage3_maskfix_long_core_gpu2.ncu-rep"}
OUTPUT = ROOT / "results/research/E130/ncu_long_core_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage2pp-report", type=Path, default=REPORTS["sage2pp"])
    parser.add_argument("--sage3-report", type=Path, default=REPORTS["sage3_maskfix"])
    parser.add_argument("--sage3-name", default="sage3_maskfix")
    parser.add_argument("--scope", default="same real H3 [1,56,22677,128] BF16 QKV, noncausal, GPU2")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-name", default="sage3_bf16_delta")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    reports = {"sage2pp": args.sage2pp_report,
               args.sage3_name: args.sage3_report}
    if args.candidate is not None:
        reports[args.candidate_name] = args.candidate
    summary = {}
    for name, path in reports.items():
        report = read_ncu(path)
        fields = report["fields"]
        summary[name] = {
            alias: {"value": fields[key]["value"], "unit": fields[key]["unit"]}
            if key in fields else None for alias, key in METRICS.items()
        }
        summary[name].update(report_sha256=report["sha256"],
                             report_path=report["path"],
                             raw_metric_count=report["metric_count"])
        if float(summary[name]["sm_count"]["value"]) != 110:
            raise RuntimeError("not RTX PRO 5000")
    result = {"format": "e130-long-sage2pp-sage3-core-ncu-v1",
              "scope": args.scope,
              "caveat": "NCU replay perturbs duration; use CUDA events for speed. Stall ratios overlap and are not additive.",
              "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {key: row[key]["value"] for key in
                             ("tensor_pipe_pct_elapsed", "dynamic_shared_kb",
                              "registers_per_thread", "spilling_requests",
                              "warp_stall_sleeping_per_issue", "warp_stall_short_scoreboard_per_issue")}
                      for name, row in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
