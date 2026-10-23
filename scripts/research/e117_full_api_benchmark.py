#!/usr/bin/env python3
"""CUDA-event full-API timing: fixed Sage3 and dense/sparse native mixed."""
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


def time_cuda(fn, prepare, warmup=20, batches=8, repeats=12):
    for _ in range(warmup):
        prepare()
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(batches):
        events = []
        for _ in range(repeats):
            prepare()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            events.append((a, b))
        torch.cuda.synchronize()
        means.append(statistics.mean(a.elapsed_time(b) for a, b in events))
    mean = statistics.mean(means)
    return {"mean_ms": mean, "cv_percent":
            statistics.stdev(means) / mean * 100,
            "batch_means_ms": means, "calls": batches * repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E117/full_api_gpu0.json")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed")]
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda as dense_router
    import sage3_mixed_router_sparse_cuda as sparse_router
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[key][:9482].to(f"cuda:{args.gpu}")
               .transpose(0, 1).unsqueeze(0) for key in ("q", "k", "v"))
    k_work = torch.empty_like(k)

    def prepare():
        k_work.copy_(k)

    def mixed(router):
        qp, kp, vp, ds = preprocess_qkv(q, k_work, v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        v8, scale8, route, _ = router.route_and_pack_v(vp, 0.10)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, 9482, None, 128 ** -0.5, False, True, True)
        return fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, scale8, route)[0][:, :, :9482]

    with torch.inference_mode():
        prepare()
        dense = mixed(dense_router)
        prepare()
        sparse = mixed(sparse_router)
        assert torch.equal(dense, sparse)
        report = {
            "format": "e117-full-api-v1", "gpu": torch.cuda.get_device_name(args.gpu),
            "capture": str(CAPTURE), "shape": list(q.shape),
            "timing_excludes_benchmark_only_k_restore": True,
            "dense_sparse_bitwise_equal": True,
            "timing": {
                "pure_fixed_online": time_cuda(lambda:
                    sageattn3_blackwell(q, k_work, v, is_causal=False,
                                       per_block_mean=True), prepare),
                "mixed_dense_online": time_cuda(lambda: mixed(dense_router), prepare),
                "mixed_sparse_online": time_cuda(lambda: mixed(sparse_router), prepare),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timing_ms": {k: v["mean_ms"] for k, v
                                    in report["timing"].items()},
                      "cv_percent": {k: v["cv_percent"] for k, v
                                     in report["timing"].items()}}, indent=2))


if __name__ == "__main__":
    main()
