#!/usr/bin/env python3
"""CUDA Graph capture/replay smoke test for Sage3 opt-in fast preprocessing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt")


def timing(fn) -> dict:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {"mean_ms": statistics.mean(elapsed),
            "median_ms": statistics.median(elapsed),
            "min_ms": min(elapsed), "max_ms": max(elapsed),
            "calls": len(elapsed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens not in (9482, 22677):
        raise ValueError("unexpected validation length")
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell

    torch.cuda.set_device(0)
    source = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = [source[name][..., :args.tokens, :].to("cuda:0").contiguous()
               for name in ("q", "k", "v")]

    def call():
        return sageattn3_blackwell(q, k, v, is_causal=False,
                                   per_block_mean=True, fast_preprocess=True)

    with torch.inference_mode():
        eager = call()
        warm_stream = torch.cuda.Stream()
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(warm_stream)
        warm_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = call()
        graph.replay()
        torch.cuda.synchronize()
        equal_first = bool(torch.equal(captured, eager))
        graph.replay()
        torch.cuda.synchronize()
        equal_second = bool(torch.equal(captured, eager))
        if not (equal_first and equal_second and torch.isfinite(captured).all()):
            raise RuntimeError("captured Sage3 differs from eager output")
        eager_timing = timing(call)
        graph_timing = timing(graph.replay)
    report = {"format": "e130-sage3-fast-api-cuda-graph-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(args.sage3_dir.resolve()),
              "capture": str(CAPTURE), "tokens": args.tokens,
              "shape_hnd": list(q.shape), "gpu": torch.cuda.get_device_name(0),
              "bitwise_equal_replay_1": equal_first,
              "bitwise_equal_replay_2": equal_second,
              "eager": eager_timing, "graph": graph_timing}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tokens": args.tokens, "equal": equal_first and equal_second,
                      "eager_ms": eager_timing["mean_ms"],
                      "graph_ms": graph_timing["mean_ms"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
