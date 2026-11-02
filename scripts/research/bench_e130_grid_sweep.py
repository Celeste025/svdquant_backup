#!/usr/bin/env python3
"""Sweep Sage3 persistent CTA count on a real long H3 attention call."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/sage3_h3_maskfix"))
from sageattn3 import sageattn3_blackwell

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")
OUTPUT = ROOT / "results/research/E130/sage3_grid_sweep_long_gpu2.json"
GRIDS = (55, 80, 96, 110, 128, 165, 220)


def main() -> None:
    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = [payload[name].to("cuda:0") for name in ("q", "k", "v")]
    if any(t.shape != (1, 56, 22677, 128) for t in (q, k, v)):
        raise RuntimeError("unexpected H3 QKV shape")
    k_work = torch.empty_like(k)

    def invoke(grid: int):
        os.environ["SAGE3_GRID_CTAS"] = str(grid)
        k_work.copy_(k)
        return sageattn3_blackwell(q, k_work, v, is_causal=False, per_block_mean=True)

    with torch.inference_mode():
        reference = invoke(110).clone()
        checks = {}
        for grid in GRIDS:
            candidate = invoke(grid)
            delta = candidate.float() - reference.float()
            checks[str(grid)] = {
                "bitwise_equal_to_110": bool(torch.equal(candidate, reference)),
                "nmse_vs_110": float(delta.square().sum() / reference.float().square().sum()),
                "finite": bool(torch.isfinite(candidate).all()),
            }
            print("check", grid, checks[str(grid)], flush=True)
        rounds = {str(grid): [] for grid in GRIDS}
        for order in (GRIDS, tuple(reversed(GRIDS))):
            for grid in order:
                os.environ["SAGE3_GRID_CTAS"] = str(grid)
                for _ in range(25):
                    k_work.copy_(k)
                    sageattn3_blackwell(q, k_work, v, is_causal=False, per_block_mean=True)
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(96)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(96)]
                for start, end in zip(starts, ends):
                    k_work.copy_(k)
                    start.record()
                    sageattn3_blackwell(q, k_work, v, is_causal=False, per_block_mean=True)
                    end.record()
                torch.cuda.synchronize()
                samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
                row = {"mean_ms": statistics.mean(samples),
                       "median_ms": statistics.median(samples),
                       "cv": statistics.stdev(samples) / statistics.mean(samples)}
                rounds[str(grid)].append(row)
                print("timing", grid, row, flush=True)
    means = {grid: statistics.mean(row["mean_ms"] for row in entries)
             for grid, entries in rounds.items()}
    report = {"format": "e130-sage3-grid-sweep-v1", "gpu": torch.cuda.get_device_name(),
              "capture": str(CAPTURE), "shape_hnd": list(q.shape),
              "scope": "full online Sage3 API; K restoration outside CUDA event",
              "default_grid": 110, "checks": checks, "rounds": rounds, "means_ms": means}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("best", min(means, key=means.get), means, flush=True)


if __name__ == "__main__":
    main()
