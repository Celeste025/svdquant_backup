#!/usr/bin/env python3
"""Cost split for fixed Sage3, mixed-with-no-FP8, actual mixed, and V router."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def time_cuda(fn, warmup: int = 20, batches: int = 8, repeats: int = 12) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(batches):
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        means.append(statistics.mean(start.elapsed_time(end)
                                     for start, end in events))
    mean = statistics.mean(means)
    return {"mean_ms": mean, "cv_percent":
            statistics.stdev(means) / mean * 100,
            "batch_means_ms": means, "calls": batches * repeats}


def nmse(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    return float((x - y).square().sum() / y.square().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E117/cost_split_gpu0.json")
    args = parser.parse_args()

    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed")]
    import fp4attn_cuda
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    with torch.inference_mode():
        q, k, v = (payload[name][:9482].to(f"cuda:{args.gpu}")
                   .transpose(0, 1).unsqueeze(0) for name in ("q", "k", "v"))
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, 9482, None, 128 ** -0.5, False, True, True)
        v8, s8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        no_route = torch.zeros_like(route)
        pure = fp4attn_cuda.fwd(*packet)[0][:, :, :9482]
        mixed_zero = fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, s8, no_route)[0][:, :, :9482]
        mixed_real = fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, s8, route)[0][:, :, :9482]
        per_head = route.sum(dim=-1).flatten().tolist()
        report = {
            "format": "e117-native-mixed-cost-split-v1",
            "capture": str(CAPTURE),
            "gpu": torch.cuda.get_device_name(args.gpu),
            "selected_tiles": int(route.sum()),
            "total_tiles": route.numel(),
            "heads_with_fp8": sum(x > 0 for x in per_head),
            "fp8_tiles_per_head": per_head,
            "nmse_zero_vs_pure": nmse(mixed_zero, pure),
            "nmse_real_vs_pure": nmse(mixed_real, pure),
            "timing": {
                "pure_core": time_cuda(lambda: fp4attn_cuda.fwd(*packet)),
                "mixed_zero_core": time_cuda(lambda:
                    fp4attn_mixed_maskfix_cuda.fwd_mixed(
                        *packet, v8, s8, no_route)),
                "mixed_real_core": time_cuda(lambda:
                    fp4attn_mixed_maskfix_cuda.fwd_mixed(
                        *packet, v8, s8, route)),
                "router_dense": time_cuda(lambda:
                    sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)),
            },
        }
    assert report["nmse_zero_vs_pure"] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_tiles": report["selected_tiles"],
                      "heads_with_fp8": report["heads_with_fp8"],
                      "nmse_zero_vs_pure": report["nmse_zero_vs_pure"],
                      "timing_ms": {name: value["mean_ms"] for name, value
                                    in report["timing"].items()}}, indent=2))


if __name__ == "__main__":
    main()
