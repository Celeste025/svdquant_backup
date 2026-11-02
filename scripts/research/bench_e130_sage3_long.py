#!/usr/bin/env python3
"""Paired online latency for official and fused Sage3 on long real H3 QKV."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/research"), str(ROOT / "third_party/sage3_h3_maskfix")]

from bench_h3_sage_two_scope import measure
from e130_sage3_preprocess import attention

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")
OUTPUT = ROOT / "results/research/E130/preprocess_long_ab_gpu2.json"


def main() -> None:
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = [payload[name].to("cuda:0") for name in ("q", "k", "v")]
    if any(x.shape != (1, 56, 22677, 128) for x in (q, k, v)):
        raise RuntimeError("unexpected H3 QKV shape")
    k_work = torch.empty_like(k)

    def call(variant: str):
        if variant == "official":
            return sageattn3_blackwell(q, k_work, v,
                                       is_causal=False, per_block_mean=True)
        return attention(q, k_work, v, api)

    timings = {name: [] for name in ("official", "fused_preprocess")}
    with torch.inference_mode():
        for ordering in (("official", "fused_preprocess"),
                         ("fused_preprocess", "official")):
            for name in ordering:
                result = measure(lambda name=name: call(name),
                                 prepare=lambda: k_work.copy_(k))
                timings[name].append(result)
                print(name, result["mean_ms"], "ms", flush=True)
    means = {name: statistics.mean(row["mean_ms"] for row in entries)
             for name, entries in timings.items()}
    report = {"format": "e130-sage3-long-online-ab-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "capture": str(CAPTURE), "shape_hnd": list(q.shape),
              "gpu": torch.cuda.get_device_name(),
              "scope": "full online Sage3 API; K restore excluded, all preprocessing included",
              "means_ms": means, "speedup": means["official"] / means["fused_preprocess"],
              "rounds": timings}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print("speedup", report["speedup"], flush=True)


if __name__ == "__main__":
    main()
