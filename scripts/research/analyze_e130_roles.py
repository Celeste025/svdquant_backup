#!/usr/bin/env python3
"""Separate source-attributed Sage3 producer/consumer Nsight PC samples."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = Path(
    "/data1/models/svdquant-wjq/research/20261025/E130/"
    "sage3_delta_tailmask_long_core_gpu2.ncu-rep"
)
METRICS = ("# Samples", "stall_short_sb", "stall_sleep", "stall_wait",
           "stall_math", "stall_long_sb", "Instructions Executed")


def as_number(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def source_role(path: str, line: int) -> str:
    if path.endswith("/sageattn3/blackwell/mainloop_tma_ws.h"):
        if 435 <= line < 574:
            return "producer_tma_mainloop"
        if 576 <= line <= 940:
            return "consumer_mma_mainloop"
    if path.endswith("/sageattn3/blackwell/softmax_fused.h"):
        return "consumer_softmax"
    if path.endswith("/sageattn3/blackwell/epilogue_tma_ws.h"):
        if 141 <= line < 163:
            return "consumer_epilogue_fragment"
        if 163 <= line <= 216:
            return "producer_tma_epilogue"
    if path.endswith("/sageattn3/blackwell/kernel_ws.h"):
        if 140 <= line < 168:
            return "producer_kernel_control"
        if 168 <= line <= 212:
            return "consumer_kernel_control"
    if "/sageattn3/blackwell/" in path:
        return "project_shared_or_unknown"
    return "external_inline_or_unknown"


def ncu_csv(report: Path, view: str) -> str:
    return subprocess.check_output(
        ["ncu", "--import", str(report), "--page", "source",
         "--print-source", view, "--csv"], text=True
    )


def source_rows(report: Path) -> list[dict]:
    path = None
    header = None
    rows = []
    for values in csv.reader(StringIO(ncu_csv(report, "cuda,sass"))):
        if not values:
            continue
        if values[0] == "File Path":
            path = values[1]
            header = None
        elif values[0] == "Line No":
            header = values
        elif header and path and values[0].isdigit() and len(values) == len(header):
            row = dict(zip(header, values))
            samples = as_number(row.get("# Samples", "0"))
            if samples == 0:
                continue
            line = int(row["Line No"])
            rows.append({"file": path, "line": line,
                         "source": row.get("Source", "").strip(),
                         "role": source_role(path, line),
                         **{key: as_number(row.get(key, "0")) for key in METRICS}})
    return rows


def sass_total(report: Path) -> dict[str, float]:
    reader = csv.reader(StringIO(ncu_csv(report, "sass")))
    next(reader)
    header = next(reader)
    totals = defaultdict(float)
    for values in reader:
        if len(values) != len(header):
            continue
        row = dict(zip(header, values))
        for key in METRICS:
            totals[key] += as_number(row.get(key, "0"))
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E130/role_samples_gpu2.json")
    args = parser.parse_args()
    rows = source_rows(args.report)
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for key in METRICS:
            grouped[row["role"]][key] += row[key]
    source_total = {key: sum(row[key] for row in rows) for key in METRICS}
    all_sass = sass_total(args.report)
    result = {
        "format": "e130-role-attributed-source-samples-v1",
        "report": str(args.report.resolve()),
        "caveat": ("CUDA line attribution is not exact warp-ID attribution; inlined "
                   "CUTLASS code and some control instructions are unknown. "
                   "PC samples/stalls are not exclusive elapsed time. Compare "
                   "source totals with SASS totals before using proportions."),
        "source_rows": len(rows),
        "sass_totals": all_sass,
        "source_totals": source_total,
        "source_over_sass_samples": source_total["# Samples"] / all_sass["# Samples"],
        "roles": {name: dict(values) for name, values in grouped.items()},
        "top_lines_by_samples": sorted(rows, key=lambda row: row["# Samples"],
                                       reverse=True)[:35],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_over_sass_samples": result["source_over_sass_samples"],
                      "roles": result["roles"]}, indent=2))


if __name__ == "__main__":
    main()
