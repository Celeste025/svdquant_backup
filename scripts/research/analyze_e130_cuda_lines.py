#!/usr/bin/env python3
"""Rank Sage3 source lines by NCU PC-sampled stalls on long H3 attention."""
from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/data1/models/svdquant-wjq/research/20261025/E130/sage3_long_source_gpu2.ncu-rep")
OUTPUT = ROOT / "results/research/E130/sage3_long_cuda_line_hotspots.json"


def count(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-marker", default="sage3_h3_maskfix/sageattn3")
    args = parser.parse_args()
    data = subprocess.check_output([
        "ncu", "--import", str(args.report), "--page", "source",
        "--print-source", "cuda,sass", "--csv",
    ], text=True)
    current_file = None
    headers = None
    rows = []
    for values in csv.reader(StringIO(data)):
        if not values:
            continue
        if values[0] == "File Path":
            current_file = values[1]
            headers = None
        elif values[0] == "Line No":
            headers = values
        elif headers and current_file and args.source_marker in current_file:
            if not values[0].isdigit() or len(values) != len(headers):
                continue
            header_index = {name: i for i, name in enumerate(headers)}
            row = {"file": current_file, "line": int(values[0]),
                   "source": values[1].strip()}
            for key in ("# Samples", "stall_short_sb", "stall_sleep",
                        "stall_math", "stall_wait", "Instructions Executed"):
                row[key] = count(values[header_index[key]]) if key in header_index else 0.0
            if row["# Samples"]:
                rows.append(row)
    ranked = {key: sorted(rows, key=lambda row: row[key], reverse=True)[:20]
              for key in ("# Samples", "stall_short_sb", "stall_sleep",
                          "stall_math", "stall_wait")}
    result = {"format": "e130-sage3-long-cuda-line-hotspots-v1",
              "report": str(args.report), "source_rows": len(rows),
              "caveat": "Source lines can include inlined template code; samples are attribution, not exclusive duration.",
              "top_lines": ranked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_rows": len(rows),
                      "top_short_scoreboard": ranked["stall_short_sb"][:8],
                      "top_sleeping": ranked["stall_sleep"][:8]}, indent=2))


if __name__ == "__main__":
    main()
