#!/usr/bin/env python3
"""Ensure the isolated copied Sage3 CUTLASS core reproduces installed Sage3."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/sage3_native_mixed"),
                "/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe"]
import fp4attn_cuda
import fp4attn_mixed_cuda
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)


def bench(fn):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)[15]


def main():
    capture = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
                   "h3_p1_s0_b25_480x832_73f_qkv.pt")
    payload = torch.load(capture, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:9482].transpose(0, 1).unsqueeze(0)
               .contiguous().cuda() for name in ("q", "k", "v"))
    with torch.inference_mode():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        inputs = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1], ds,
                  9482, None, 128 ** -0.5, False, True, True)
        reference = fp4attn_cuda.fwd(*inputs)[0]
        fork = fp4attn_mixed_cuda.fwd(*inputs)[0]
        nmse = ((reference.float() - fork.float()).square().sum() /
                reference.float().square().sum()).item()
        if nmse != 0.0:
            raise RuntimeError(f"copied native Sage3 baseline differs: {nmse}")
        row = {"shape": list(q.shape), "fork_vs_installed_nmse": nmse,
               "installed_core_median_ms": bench(lambda: fp4attn_cuda.fwd(*inputs)),
               "fork_core_median_ms": bench(lambda: fp4attn_mixed_cuda.fwd(*inputs)),
               "warmup": 8, "repeats": 30}
        path = ROOT / "results/research/E114/sage3_core_copy_baseline.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
