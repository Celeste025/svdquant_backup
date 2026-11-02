#!/usr/bin/env python3
"""Compare BF16, Sage2++, Sage3 and exact fused Sage3 on 22,677 H3 tokens."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research"),
                str(ROOT / "third_party/sage3_h3_maskfix"),
                "/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014"]
from bench_h3_sage_real_gpu0 import metrics
from bench_h3_sage_two_scope import measure
from e130_sage3_preprocess import attention

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")
OUTPUT = ROOT / "results/research/E130/sage_long_four_arm_gpu2.json"
ARMS = ("bf16_sdpa", "sage2pp", "sage3_fixed", "sage3_exact_fused")


def main() -> None:
    from sageattention import sageattn
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as sage_api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = [payload[name].to("cuda:0") for name in ("q", "k", "v")]
    if any(x.shape != (1, 56, 22677, 128) for x in (q, k, v)):
        raise RuntimeError("unexpected captured shape")
    n = q.shape[2]
    k_work = torch.empty_like(k)

    def call(arm: str):
        if arm == "bf16_sdpa":
            return F.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5)
        if arm == "sage2pp":
            return sageattn(q, k, v, tensor_layout="HND", is_causal=False,
                            sm_scale=128 ** -0.5)
        if arm == "sage3_fixed":
            return sageattn3_blackwell(q, k_work, v,
                                       is_causal=False, per_block_mean=True)
        return attention(q, k_work, v, sage_api, exact_delta=True,
                         contiguous_output=True)

    with torch.inference_mode():
        ref = call("bf16_sdpa")
        checks = {}
        for arm in ARMS:
            if arm.startswith("sage3"):
                k_work.copy_(k)
            result = call(arm)
            checks[arm] = metrics(result, ref)
        k_work.copy_(k)
        fixed = call("sage3_fixed")
        k_work.copy_(k)
        exact = call("sage3_exact_fused")
        if not torch.equal(exact, fixed):
            raise RuntimeError("exact fused Sage3 differs from fixed Sage3")
        timings = {arm: [] for arm in ARMS}
        for ordering in (ARMS, tuple(reversed(ARMS))):
            for arm in ordering:
                prep = (lambda: k_work.copy_(k)) if arm.startswith("sage3") else (lambda: None)
                row = measure(lambda arm=arm: call(arm), prepare=prep)
                timings[arm].append(row)
                print(arm, row["mean_ms"], "ms", flush=True)
    means = {arm: statistics.mean(row["mean_ms"] for row in rows)
             for arm, rows in timings.items()}
    ops = 4 * 56 * n * n * 128
    report = {"format": "e130-long-four-arm-attention-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "capture": str(CAPTURE), "shape_hnd": list(q.shape),
              "gpu": torch.cuda.get_device_name(),
              "scope": "online attention API; Sage3 disposable K restored before start event; QKV layout already HND",
              "operations": ops, "checks_vs_bf16": checks,
              "sage3_exact_bitwise_equal": True,
              "means_ms": means,
              "tops": {arm: ops / 1e12 / (value / 1000) for arm, value in means.items()},
              "rounds": timings}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"means_ms": means, "tops": report["tops"]}, indent=2))


if __name__ == "__main__":
    main()
