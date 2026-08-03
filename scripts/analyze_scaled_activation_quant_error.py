#!/usr/bin/env python3
"""Compare post-SVDQuant-scaling activation outliers and INT4 error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from quant_error_metrics import quant_metrics  # noqa: F401 — re-exported for callers


def analyze_wan(args: argparse.Namespace) -> None:
    root = args.wan_calib_dir
    manifest = json.loads((root / "manifest.json").read_text())
    checkpoint = torch.load(args.wan_checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state"]
    reports = []
    for record in manifest["groups"]:
        entries = [state[name] for name in record["names"]]
        smooth = entries[0]["smooth"].float()
        unsigned = bool(entries[0]["unsigned_activation"].item())
        shift = float(entries[0]["input_shift"].item())
        shape = tuple(record["shape"])
        raw = np.memmap(root / record["path"], dtype=np.uint16, mode="r", shape=shape)
        x = torch.from_numpy(raw).view(torch.bfloat16)
        count = min(args.rows_per_layer, shape[0])
        indices = torch.linspace(0, shape[0] - 1, count).long()
        sample = x[indices].float()
        if shift:
            sample.add_(shift)
        metrics = quant_metrics(sample, smooth, unsigned)
        metrics.update(
            {
                "model": "Wan2.1-1.3B",
                "group_index": record["group_index"],
                "names": record["names"],
                "rows": count,
            }
        )
        reports.append(metrics)
        print(f"Wan group {record['group_index']:03d}/{len(manifest['groups'])}: nmse={metrics['nmse']:.6g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("wan",), required=True)
    parser.add_argument(
        "--wan-calib-dir",
        type=Path,
        default=Path("outputs/wan_svdquant_calib_large"),
    )
    parser.add_argument(
        "--wan-checkpoint",
        type=Path,
        default=Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt"),
    )
    parser.add_argument("--rows-per-layer", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze_wan(args)


if __name__ == "__main__":
    main()
