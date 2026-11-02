#!/usr/bin/env python3
"""Aggregate Nsight Compute source-counter SASS samples by instruction kind."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from io import StringIO
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/data1/models/svdquant-wjq/research/20261025/E130")


def number(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=RAW / "sage3_long_source_gpu2.ncu-rep")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E130/sage3_long_sass_hotspots.json")
    args = parser.parse_args()
    data = subprocess.check_output([
        "ncu", "--import", str(args.report), "--page", "source",
        "--print-source", "sass", "--csv",
    ], text=True)
    reader = csv.reader(StringIO(data))
    kernel = next(reader)[1]
    columns = next(reader)
    grouped = defaultdict(lambda: defaultdict(float))
    addresses = []
    for values in reader:
        if len(values) != len(columns):
            continue
        row = dict(zip(columns, values))
        source = row["Source"].strip()
        tokens = [token for token in source.split() if not token.startswith("@")]
        match = re.match(r"([A-Z][A-Z0-9_.]*)", tokens[0]) if tokens else None
        op = match.group(1).split(".")[0] if match else "UNKNOWN"
        selected = {key: number(row.get(key, "0")) for key in
                    ("# Samples", "Instructions Executed", "stall_short_sb",
                     "stall_sleep", "stall_math", "stall_wait", "stall_long_sb")}
        for key, value in selected.items():
            grouped[op][key] += value
        addresses.append({"address": row["Address"], "sass": source, **selected})
    ranked = {key: sorted(({"opcode": op, "value": metrics[key]} for op, metrics in grouped.items()),
                          key=lambda row: row["value"], reverse=True)[:15]
              for key in ("# Samples", "Instructions Executed", "stall_short_sb",
                          "stall_sleep", "stall_math", "stall_wait", "stall_long_sb")}
    result = {"format": "e130-sass-source-counter-hotspots-v1",
              "report": str(args.report), "kernel": kernel,
              "sampled_sass_instructions": len(addresses),
              "caveat": "PC sampling attribution identifies stall locations, not exact exclusive time; opcode categories overlap and have warp-specialized roles.",
              "top_opcode": ranked,
              "top_addresses_by_samples": sorted(addresses, key=lambda row: row["# Samples"], reverse=True)[:30],
              "top_addresses_by_sleeping": sorted(addresses, key=lambda row: row["stall_sleep"], reverse=True)[:30],
              "top_addresses_by_short_scoreboard": sorted(addresses, key=lambda row: row["stall_short_sb"], reverse=True)[:30]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sampled_sass_instructions": len(addresses),
                      "top_samples": ranked["# Samples"][:8],
                      "top_short_scoreboard": ranked["stall_short_sb"][:8],
                      "top_sleeping": ranked["stall_sleep"][:8]}, indent=2))


if __name__ == "__main__":
    main()
