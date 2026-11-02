#!/usr/bin/env python3
"""Paired CUDA-event timing for Sage3 Q-center plus FP4-pack fusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

from bench_h3_sage_two_scope import measure

CAPTURE = Path(
    "/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/"
    "race/block25_qkv_h56.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    n = args.tokens
    if payload["q"].ndim == 3:
        q, k, v = [payload[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
                   for name in ("q", "k", "v")]
    else:
        q, k, v = [payload[name][:, :, :n].to("cuda:0") for name in ("q", "k", "v")]
    b, h, _, d = q.shape
    p = ((n + 127) // 128) * 128
    qp = torch.empty((b, h, p, d), device=q.device, dtype=q.dtype)
    qm = torch.empty((b, h, p // 128, d), device=q.device, dtype=q.dtype)
    qm_only = torch.empty_like(qm)
    qcode = torch.empty((b, h, p, d // 2), device=q.device, dtype=torch.uint8)
    qscale = torch.empty((b, h, p, d // 16), device=q.device, dtype=torch.float8_e4m3fn)

    def old_pack() -> None:
        api._e130_group_mean_pad_q[(b, h, p // 128)](
            q, qp, qm, n, d, *q.stride(), *qp.stride(), *qm.stride(), num_warps=4
        )
        api.fp4quant_cuda.scaled_fp4_quant(qp, qcode, qscale, 1)

    def fused_pack() -> None:
        api._e130_group_mean_q_only[(b, h, p // 128)](
            q, qm_only, n, d, *q.stride(), *qm_only.stride(), num_warps=4
        )
        api.fp4quant_cuda.scaled_fp4_quant_group_center(q, qm_only, qcode, qscale)

    def old_online() -> torch.Tensor:
        return sageattn3_blackwell(q, k, v, fast_preprocess=True)

    def fused_online() -> torch.Tensor:
        return sageattn3_blackwell(q, k, v, fast_preprocess=True, fast_qpack=True)

    def combined_online() -> torch.Tensor:
        return sageattn3_blackwell(q, k, v, fast_preprocess=True,
                                   fast_qpack=True, fast_delta_gemm=True)

    with torch.inference_mode():
        original = old_online()
        candidate = fused_online()
        combined = combined_online()
        if not torch.equal(original, candidate) or not torch.equal(original, combined):
            raise RuntimeError("fused-Q attention output differs from baseline")
        variants = {
            "old_pack": old_pack, "fused_pack": fused_pack,
            "old_online": old_online, "fused_online": fused_online,
            "combined_online": combined_online,
        }
        timings = {name: [] for name in variants}
        for ordering in (tuple(variants), tuple(reversed(variants))):
            for name in ordering:
                row = measure(variants[name])
                timings[name].append(row)
                print(f"{name}: {row['mean_ms']:.4f} ms", flush=True)
    means = {name: statistics.mean(row["mean_ms"] for row in rows)
             for name, rows in timings.items()}
    report = {
        "format": "e130-qpack-fusion-benchmark-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.sage3_dir.resolve()),
        "capture": str(args.capture.resolve()),
        "shape_hnd": list(q.shape),
        "gpu": torch.cuda.get_device_name(0),
        "output_bitwise_equal": True,
        "scope": "CUDA events; original Q center/pad+pack versus mean-only+fused pack; full online attention includes all preprocessing and core",
        "means_ms": means,
        "pack_speedup": means["old_pack"] / means["fused_pack"],
        "online_speedup": means["old_online"] / means["fused_online"],
        "combined_online_speedup": means["old_online"] / means["combined_online"],
        "rounds": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"means_ms": means, "pack_speedup": report["pack_speedup"],
                      "online_speedup": report["online_speedup"],
                      "combined_online_speedup": report["combined_online_speedup"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
