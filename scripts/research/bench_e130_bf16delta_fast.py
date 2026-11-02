#!/usr/bin/env python3
"""Compare public Sage3 with exact fast preprocessing for either delta_s dtype."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research")]

from bench_h3_sage_two_scope import measure
from e130_sage3_preprocess import attention
from e130_sage3_async_preprocess import AsyncSage3Prepacker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--fuse-k-center-pad", action="store_true")
    parser.add_argument("--fuse-k-center-pack", action="store_true")
    parser.add_argument("--async-preprocess", action="store_true")
    parser.add_argument("--mask-padded-delta", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    if payload["q"].ndim == 3:
        n = 9482
        q, k, v = [payload[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
                   for name in ("q", "k", "v")]
    elif payload["q"].ndim == 4:
        q, k, v = [payload[name].to("cuda:0") for name in ("q", "k", "v")]
        n = q.shape[2]
    else:
        raise ValueError("unsupported QKV rank")
    if q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("unexpected H3 QKV shape")
    k_work = torch.empty_like(k)
    async_packer = AsyncSage3Prepacker() if args.async_preprocess else None
    results: dict[str, dict] = {}

    def public():
        return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                   per_block_mean=True)

    def exact_fast():
        return attention(q, k_work, v, api, exact_delta=True,
                         delta_dtype=(torch.bfloat16 if args.delta_dtype == "bf16"
                                      else torch.float32),
                         mask_padded_delta=args.mask_padded_delta)

    def exact_fast_fused_k():
        return attention(q, k_work, v, api, exact_delta=True,
                         delta_dtype=(torch.bfloat16 if args.delta_dtype == "bf16"
                                      else torch.float32),
                         fuse_k_center_pad=True,
                         mask_padded_delta=args.mask_padded_delta)

    def exact_fast_fused_k_pack():
        return attention(q, k_work, v, api, exact_delta=True,
                         delta_dtype=(torch.bfloat16 if args.delta_dtype == "bf16"
                                      else torch.float32),
                         fuse_k_center_pack=True,
                         mask_padded_delta=args.mask_padded_delta)

    def exact_fast_async():
        assert async_packer is not None
        return async_packer.attention(q, k_work, v, api)

    functions = {"public": public, "exact_fast": exact_fast}
    if args.fuse_k_center_pad:
        functions["exact_fast_fused_k"] = exact_fast_fused_k
    if args.fuse_k_center_pack:
        functions["exact_fast_fused_k_pack"] = exact_fast_fused_k_pack
    if args.async_preprocess:
        functions["exact_fast_async"] = exact_fast_async
    with torch.inference_mode():
        k_work.copy_(k)
        reference = public()
        k_work.copy_(k)
        candidate = exact_fast()
        if not torch.equal(reference, candidate):
            raise RuntimeError("exact fast path is not bitwise equal to the public API")
        if args.fuse_k_center_pad:
            k_work.copy_(k)
            candidate_fused = exact_fast_fused_k()
            if not torch.equal(reference, candidate_fused):
                raise RuntimeError("fused K path is not bitwise equal to the public API")
        if args.fuse_k_center_pack:
            k_work.copy_(k)
            candidate_fused_pack = exact_fast_fused_k_pack()
            if not torch.equal(reference, candidate_fused_pack):
                raise RuntimeError("fused K pack path is not bitwise equal to the public API")
        if args.async_preprocess:
            k_work.copy_(k)
            candidate_async = exact_fast_async()
            if not torch.equal(reference, candidate_async):
                raise RuntimeError("async prepack path is not bitwise equal to the public API")
        for ordering in (list(functions), list(reversed(functions))):
            for name in ordering:
                row = measure(functions[name], prepare=lambda: k_work.copy_(k))
                results.setdefault(name, {"rounds": []})["rounds"].append(row)
                print(name, row["mean_ms"], row["cv_batch_mean_percent"], flush=True)
    for value in results.values():
        value["mean_ms"] = statistics.mean(row["mean_ms"] for row in value["rounds"])
    results["exact_fast"]["speedup_vs_public"] = (
        results["public"]["mean_ms"] /
        results["exact_fast"]["mean_ms"]
    )
    if args.fuse_k_center_pad:
        results["exact_fast_fused_k"]["speedup_vs_public"] = (
            results["public"]["mean_ms"] /
            results["exact_fast_fused_k"]["mean_ms"])
        results["exact_fast_fused_k"]["speedup_vs_exact_fast"] = (
            results["exact_fast"]["mean_ms"] /
            results["exact_fast_fused_k"]["mean_ms"])
    if args.fuse_k_center_pack:
        results["exact_fast_fused_k_pack"]["speedup_vs_public"] = (
            results["public"]["mean_ms"] /
            results["exact_fast_fused_k_pack"]["mean_ms"])
        results["exact_fast_fused_k_pack"]["speedup_vs_exact_fast"] = (
            results["exact_fast"]["mean_ms"] /
            results["exact_fast_fused_k_pack"]["mean_ms"])
    if args.async_preprocess:
        results["exact_fast_async"]["speedup_vs_public"] = (
            results["public"]["mean_ms"] /
            results["exact_fast_async"]["mean_ms"])
        if args.fuse_k_center_pad:
            results["exact_fast_async"]["speedup_vs_fused_k"] = (
                results["exact_fast_fused_k"]["mean_ms"] /
                results["exact_fast_async"]["mean_ms"])
    report = {
        "format": "e130-bf16delta-fast-ab-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "sage3_source": str(args.sage3_dir.resolve()),
        "delta_dtype": args.delta_dtype,
        "fuse_k_center_pad": args.fuse_k_center_pad,
        "fuse_k_center_pack": args.fuse_k_center_pack,
        "async_preprocess": args.async_preprocess,
        "mask_padded_delta": args.mask_padded_delta,
        "capture": str(args.capture.resolve()),
        "shape_hnd": list(q.shape),
        "bitwise_equal": True,
        "timing": "CUDA events, 25 warmups and 12x12 calls per round, reverse-order repeat; K restore outside timing",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
