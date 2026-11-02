#!/usr/bin/env python3
"""Compare Sage3 block/global Q-centering on one real H3 attention input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/research")]
from bench_h3_sage_real_gpu0 import metrics
from bench_h3_sage_two_scope import measure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    if payload["q"].ndim == 3:
        n = 9482
        q, k, v = [payload[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
                   for name in ("q", "k", "v")]
    elif payload["q"].ndim == 4:
        q, k, v = [payload[name].to("cuda:0") for name in ("q", "k", "v")]
        n = q.shape[2]
    else:
        raise ValueError("unexpected capture rank")
    if q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("unexpected QKV shape")
    k_work = torch.empty_like(k)
    rows: dict[str, dict] = {}

    def run(block_mean: bool) -> torch.Tensor:
        return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                   per_block_mean=block_mean)

    with torch.inference_mode():
        reference = F.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5)
        k_work.copy_(k)
        block_output = run(True)
        k_work.copy_(k)
        global_output = run(False)
        quality = {
            "block_vs_bf16": metrics(block_output, reference),
            "global_vs_bf16": metrics(global_output, reference),
            "global_vs_block": metrics(global_output, block_output),
            "all_finite": bool(torch.isfinite(block_output).all()
                               and torch.isfinite(global_output).all()),
        }
        if not quality["all_finite"]:
            raise RuntimeError("nonfinite attention output")
        for order in ((True, False), (False, True)):
            for block_mean in order:
                arm = "block_mean" if block_mean else "global_mean"
                timing = measure(lambda mode=block_mean: run(mode),
                                 prepare=lambda: k_work.copy_(k))
                rows.setdefault(arm, {"timings": []})["timings"].append(timing)
                print(arm, timing["mean_ms"], flush=True)
    for row in rows.values():
        row["mean_ms"] = statistics.mean(t["mean_ms"] for t in row["timings"])
    report = {
        "format": "e130-sage3-q-centering-tradeoff-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "capture": str(args.capture.resolve()),
        "sage3_dir": str(args.sage3_dir.resolve()),
        "shape_hnd": list(q.shape),
        "gpu": torch.cuda.get_device_name(0),
        "timing_scope": "online API; K restore excluded from CUDA event; HND input ready",
        "quality": quality,
        "arms": rows,
        "global_speedup": rows["block_mean"]["mean_ms"] / rows["global_mean"]["mean_ms"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"quality": quality,
                      "means_ms": {name: row["mean_ms"] for name, row in rows.items()},
                      "global_speedup": report["global_speedup"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
