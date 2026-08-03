#!/usr/bin/env python3
"""Merge and audit parallel Wan SVDQuant calibration shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs/wan_svdquant_fake/calibrated_shards")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wan_svdquant_fake/svdquant_seed44_calibrated.pt")
    )
    args = parser.parse_args()

    shard_paths = sorted(args.input_dir.glob("shard_*_of_08.pt"))
    report_paths = sorted(args.input_dir.glob("shard_*_of_08.json"))
    if len(shard_paths) != 8 or len(report_paths) != 8:
        raise RuntimeError(f"expected 8 state and report shards, got {len(shard_paths)}, {len(report_paths)}")
    state = {}
    shard_metadata = None
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        metadata = {
            key: shard.get(key)
            for key in (
                "format",
                "group_size",
                "rank",
                "weight_bits",
                "activation_mode",
                "num_grids",
                "max_iters",
                "calibration_manifest",
            )
        }
        if shard_metadata is None:
            shard_metadata = metadata
        elif metadata != shard_metadata:
            raise RuntimeError(f"inconsistent shard metadata in {path}")
        overlap = set(state).intersection(shard["state"])
        if overlap:
            raise RuntimeError(f"duplicate modules in {path}: {sorted(overlap)}")
        state.update(shard["state"])
    reports = []
    for path in report_paths:
        reports.extend(json.loads(path.read_text()))
    if len(state) != 300 or len(reports) != 210:
        raise RuntimeError(f"expected 300 modules/210 groups, got {len(state)}/{len(reports)}")

    search = [r["search_nmse"] for r in reports]
    final = [r["final_nmse"] for r in reports]
    iterations = [r["iterations_run"] for r in reports]
    summary = {
        "num_modules": len(state),
        "num_groups": len(reports),
        "mean_search_nmse": sum(search) / len(search),
        "mean_final_nmse": sum(final) / len(final),
        "mean_relative_improvement": sum((a - b) / a for a, b in zip(search, final, strict=True))
        / len(search),
        "mean_iterations_run": sum(iterations) / len(iterations),
        "max_iterations_run": max(iterations),
        "groups_hit_max_iters": sum(i == 100 for i in iterations),
    }
    payload = {
        "format": "wan-svdquant-calibrated-v1",
        "group_size": shard_metadata["group_size"],
        "rank": shard_metadata["rank"],
        "weight_bits": shard_metadata["weight_bits"] or 4,
        "activation_mode": shard_metadata["activation_mode"] or "w4a4",
        "num_grids": shard_metadata["num_grids"],
        "max_iters": shard_metadata["max_iters"],
        "calibration_manifest": shard_metadata["calibration_manifest"],
        "state": state,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "groups": reports}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
