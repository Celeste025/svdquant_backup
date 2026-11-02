#!/usr/bin/env python3
"""E113 feasibility check: real H3 QKV, official Sage3 vs mixed Triton P/V."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
sys.path.insert(0, str(SAGE))

from sageattn3 import sageattn3_blackwell
from e113_mixed_attention import mixed_sage3_attention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert args.capture.is_file() and SAGE.is_dir()
    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    assert payload["cu_seqlens"].tolist() == [0, 9482, 9536]
    q, k, v = (payload[name][:9482].transpose(0, 1).unsqueeze(0).contiguous().cuda()
               for name in ("q", "k", "v"))
    def sage():
        return sageattn3_blackwell(q, k.clone(), v, is_causal=False,
                                   per_block_mean=True)

    def mixed():
        return mixed_sage3_attention(q, k, v, threshold=0.10)

    report = {"format": "e113-h3-mixed-attention-preflight-v1",
              "capture": str(args.capture), "shape": list(q.shape),
              "gpu": torch.cuda.get_device_name(), "arms": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, fn in (("sage3", sage), ("mixed_triton", mixed)):
        begin = time.monotonic()
        print(f"warmup {name}", flush=True)
        out = fn()
        torch.cuda.synchronize()
        compile_or_warmup_seconds = time.monotonic() - begin
        assert out.shape == q.shape and torch.isfinite(out).all().item()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        for start, end in zip(starts, ends):
            start.record()
            next_out = fn()
            end.record()
        torch.cuda.synchronize()
        ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        row = {"warmup_or_compile_seconds": compile_or_warmup_seconds,
               "online_ms": ms, "online_mean_ms": sum(ms) / len(ms),
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
        report["arms"][name] = row
        print(json.dumps({name: row}, indent=2), flush=True)
        if name == "sage3":
            baseline = out.float()
        else:
            report["mixed_vs_sage3_nmse"] = float(
                (out.float() - baseline).square().sum() / baseline.square().sum())
        del out, next_out
    ratio = report["arms"]["mixed_triton"]["online_mean_ms"] / report["arms"]["sage3"]["online_mean_ms"]
    report["mixed_over_sage3_time_ratio"] = ratio
    # One 20-step H3 generation performs 50 main-attention calls per step.
    report["estimated_mixed_attention_seconds_per_video_at_9482_tokens"] = (
        report["arms"]["mixed_triton"]["online_mean_ms"] * 1000 / 1000)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"preflight report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
