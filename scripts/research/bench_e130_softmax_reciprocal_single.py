#!/usr/bin/env python3
"""Benchmark one Sage3 binary; invoke in separate processes for an A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path("/tmp/e130-softmax-baseline")
SOURCE_DIR = ROOT / "third_party/sage3_h3_maskfix"
CAPTURES = {
    "short": Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt"),
    "long": Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt"),
}


def one_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, call) -> dict:
    k_work = torch.empty_like(k)
    with torch.inference_mode():
        k_work.copy_(k)
        output = call(q, k_work, v, is_causal=False, per_block_mean=True)
        if not torch.isfinite(output).all():
            raise RuntimeError("nonfinite output")
        digest = hashlib.sha256(output.contiguous().cpu().view(torch.uint16).numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        for _ in range(25):
            k_work.copy_(k)
            call(q, k_work, v, is_causal=False, per_block_mean=True)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(96)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(96)]
        for start, end in zip(starts, ends):
            k_work.copy_(k)
            start.record()
            call(q, k_work, v, is_causal=False, per_block_mean=True)
            end.record()
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {"shape": list(q.shape), "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times), "cv": statistics.stdev(times) / statistics.mean(times),
            "output_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("baseline", "reciprocal", "twoacc", "n64"))
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    binary_dir = BASELINE_DIR if args.variant == "baseline" else SOURCE_DIR
    sys.path[:0] = [str(binary_dir), str(SOURCE_DIR)]
    from sageattn3 import sageattn3_blackwell
    import fp4attn_cuda
    if Path(fp4attn_cuda.__file__).parent.resolve() != binary_dir.resolve():
        raise RuntimeError(f"loaded wrong binary: {fp4attn_cuda.__file__}")
    torch.cuda.set_device(0)
    results = {}
    for name, path in CAPTURES.items():
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if name == "short":
            q, k, v = [payload[key].to("cuda:0").transpose(0, 1).unsqueeze(0)
                       for key in ("q", "k", "v")]
        else:
            q, k, v = [payload[key].to("cuda:0") for key in ("q", "k", "v")]
        results[name] = one_shape(q, k, v, sageattn3_blackwell)
        print(name, json.dumps(results[name]), flush=True)
        del q, k, v, payload
    report = {"format": "e130-softmax-reciprocal-single-process-v1",
              "variant": args.variant, "round": args.round, "binary": fp4attn_cuda.__file__,
              "gpu": torch.cuda.get_device_name(), "scope": "full Sage3 API, K restoration outside event",
              "results": results}
    output_path = ROOT / f"results/research/E130/softmax_reciprocal_{args.variant}_r{args.round}_gpu2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
