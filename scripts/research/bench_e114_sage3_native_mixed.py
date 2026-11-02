#!/usr/bin/env python3
"""Paired real-H3 correctness and CUDA-event profile for native Sage3 mixed PV."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")
sys.path[:0] = [str(ROOT / "third_party/sage3_native_mixed"),
                str(ROOT / "scripts/research"), str(SAGE)]

import fp4attn_cuda
import fp4attn_mixed_cuda
import sage3_mixed_router_cuda
from e113_native_mixed_attention import native_mixed_sage3_attention
from sageattn3 import sageattn3_blackwell
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).square().sum() /
            b.float().square().sum()).item()


def bench(fn, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values),
            "min_ms": min(values), "max_ms": max(values),
            "std_ms": statistics.pstdev(values), "warmup": warmup,
            "repeats": repeats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=9482)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E114/native_full_attention_profile.json")
    args = parser.parse_args()
    if args.tokens < 128 or args.tokens > 9482 or args.heads < 1 or args.heads > 56:
        raise ValueError("capture supports 128..9482 tokens and 1..56 heads")
    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:args.tokens].transpose(0, 1)[:args.heads]
               .unsqueeze(0).contiguous().cuda() for name in ("q", "k", "v"))

    def pack():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packed = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1], ds,
                  args.tokens, None, 128 ** -0.5, False, True, True)
        return vp, packed

    def native_full():
        vp, packed = pack()
        v8, scale8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        return fp4attn_mixed_cuda.fwd_mixed(*packed, v8, scale8, route)[0][
            :, :, :args.tokens].contiguous()

    with torch.inference_mode():
        vp, packed = pack()
        v8, scale8, route, relsq = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        official_core = fp4attn_cuda.fwd(*packed)[0][:, :, :args.tokens].contiguous()
        copied_core = fp4attn_mixed_cuda.fwd(*packed)[0][:, :, :args.tokens].contiguous()
        native = fp4attn_mixed_cuda.fwd_mixed(*packed, v8, scale8, route)[0][
            :, :, :args.tokens].contiguous()
        official = sageattn3_blackwell(q, k.clone(), v, is_causal=False,
                                      per_block_mean=True)
        triton = native_mixed_sage3_attention(q, k, v, threshold=0.10)
        outputs = (official_core, copied_core, native, official, triton)
        if not all(torch.isfinite(x).all().item() for x in outputs):
            raise RuntimeError("non-finite attention output")
        error = {
            "copied_fp4_vs_installed_core": nmse(copied_core, official_core),
            "native_mixed_vs_official_fp4": nmse(native, official_core),
            "native_mixed_vs_triton_mixed": nmse(native, triton),
            "triton_mixed_vs_official_fp4": nmse(triton, official_core),
            "official_full_vs_official_core": nmse(official, official_core),
        }
        if error["copied_fp4_vs_installed_core"] != 0:
            raise RuntimeError("copied FP4 core no longer matches official Sage3")
        if error["native_mixed_vs_official_fp4"] > 0.01:
            raise RuntimeError("native mixed route exceeds FP4 numerical gate")
        if error["native_mixed_vs_triton_mixed"] > 0.01:
            raise RuntimeError("native mixed differs too much from Triton mixed")

        timing = {}
        tasks = {
            "router_pack_v": lambda: sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10),
            "official_fp4_core": lambda: fp4attn_cuda.fwd(*packed),
            "copied_fp4_core": lambda: fp4attn_mixed_cuda.fwd(*packed),
            "native_mixed_core": lambda: fp4attn_mixed_cuda.fwd_mixed(
                *packed, v8, scale8, route),
            "official_full_api": lambda: sageattn3_blackwell(
                q, k.clone(), v, is_causal=False, per_block_mean=True),
            "native_mixed_full_api": native_full,
            "triton_mixed_full_api": lambda: native_mixed_sage3_attention(
                q, k, v, threshold=0.10),
        }
        for name, fn in tasks.items():
            timing[name] = bench(fn, args.warmup, args.repeats)
            print(name, timing[name]["median_ms"], "ms", flush=True)
        report = {
            "format": "e114-native-sage3-mixed-profile-v1",
            "capture": str(CAPTURE), "shape": list(q.shape),
            "padded_tokens": int(vp.shape[2]), "gpu": torch.cuda.get_device_name(),
            "threshold": 0.10, "selected_v_tiles": int(route.sum().item()),
            "total_v_tiles": route.numel(),
            "fp4_group": 16, "fp8_group": 128,
            "nmse": error, "timing": timing,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "triton_mma_check": native_mixed_sage3_attention.last_mma_check,
        }
        if not all(math.isfinite(x) for x in error.values()):
            raise RuntimeError("non-finite error metric")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
