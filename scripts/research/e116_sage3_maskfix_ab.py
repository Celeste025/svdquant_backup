#!/usr/bin/env python3
"""Isolated E116 correctness and CUDA-event timing for Sage3 mask A/B.

Run baseline and fixed builds in separate processes. Extension names and the
sageattn3 Python package cannot be swapped safely inside one interpreter.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")
NATIVE = ROOT / "third_party/sage3_native_mixed"


def nmse(x: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = x.float(), reference.float()
    return float((a - b).square().sum() / b.square().sum())


def time_cuda(fn, prepare=lambda: None, *, warmup=20, batches=8, repeats=12):
    for _ in range(warmup):
        prepare()
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(batches):
        samples = []
        for _ in range(repeats):
            prepare()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            samples.append((start, end))
        torch.cuda.synchronize()
        means.append(statistics.mean(a.elapsed_time(b) for a, b in samples))
    mean = statistics.mean(means)
    return {"mean_ms": mean, "median_batch_ms": statistics.median(means),
            "cv_percent": 100 * statistics.stdev(means) / mean,
            "batch_means_ms": means, "calls": batches * repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-root", type=Path, required=True)
    parser.add_argument("--mixed-module", choices=("fp4attn_mixed_cuda",
                        "fp4attn_mixed_maskfix_cuda"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-fixed", action="store_true")
    args = parser.parse_args()

    # Bind the requested pure Sage3 package/extension first. The Triton
    # experiment inserts its historical Sage path on import, but Python's
    # module cache keeps the selected package/extension used in this process.
    sys.path[:0] = [str(args.sage_root), str(NATIVE),
                    str(ROOT / "third_party/MixAttention"),
                    str(ROOT / "scripts/research")]
    import fp4attn_cuda
    import sage3_mixed_router_cuda
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)
    mixed_cuda = importlib.import_module(args.mixed_module)
    from e113_native_mixed_attention import native_mixed_sage3_attention
    from e080_pv_contract import k_permutation

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    assert payload["cu_seqlens"].tolist() == [0, 9482, 9536]
    report = {"format": "e116-sage3-maskfix-ab-v1",
              "sage_root": str(args.sage_root.resolve()),
              "pure_extension": str(Path(fp4attn_cuda.__file__).resolve()),
              "mixed_extension": str(Path(mixed_cuda.__file__).resolve()),
              "gpu": torch.cuda.get_device_name(args.gpu),
              "capture": str(CAPTURE), "cases": {}, "timings": {}}
    main_inputs = None
    main_packet = None
    with torch.inference_mode():
        for name, begin, length, heads in (
                ("control_32", 9482, 32, 4),
                ("pad_54", 9482, 54, 4),
                ("main_9482", 0, 9482, 56)):
            q, k, v = (payload[key][begin:begin + length].to(f"cuda:{args.gpu}")
                       .transpose(0, 1).unsqueeze(0) for key in ("q", "k", "v"))
            q, k, v = q[:, :heads], k[:, :heads], v[:, :heads]
            qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
            q4 = scale_and_quant_fp4(qp)
            k4 = scale_and_quant_fp4_permute(kp)
            v4 = scale_and_quant_fp4_transpose(vp)
            packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                      ds, length, None, 128 ** -0.5, False, True, True)
            pure = fp4attn_cuda.fwd(*packet)[0][:, :, :length].contiguous()
            v8, s8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
            mixed = mixed_cuda.fwd_mixed(*packet, v8, s8, route)[0][
                :, :, :length].contiguous()
            triton_pure = native_mixed_sage3_attention(
                q, k, v, threshold=0.50)
            triton_mixed = native_mixed_sage3_attention(
                q, k, v, threshold=0.10)
            compatible = native_mixed_sage3_attention(
                q, k, v, threshold=0.50, match_sage3_padding_mask=True)
            assert all(bool(torch.isfinite(x).all()) for x in
                       (pure, mixed, triton_pure, triton_mixed, compatible))
            padded = qp.shape[2]
            physical = torch.arange(padded, device=q.device)
            logical = k_permutation(padded, q.device)
            case = {
                "shape": list(q.shape), "padded_length": padded,
                "wrong_masked_valid_keys_in_old_sage3": int(
                    ((logical < length) & (physical >= length)).sum()),
                "wrong_unmasked_padding_keys_in_old_sage3": int(
                    ((logical >= length) & (physical < length)).sum()),
                "fp8_tiles": int(route.sum()), "total_tiles": route.numel(),
                "triton_pure_vs_pure_nmse": nmse(triton_pure, pure),
                "triton_legacy_mask_vs_pure_nmse": nmse(compatible, pure),
                "triton_mixed_vs_native_mixed_nmse": nmse(triton_mixed, mixed),
                "native_mixed_vs_pure_nmse": nmse(mixed, pure),
            }
            report["cases"][name] = case
            print(name, json.dumps(case), flush=True)
            if args.expect_fixed:
                assert case["triton_pure_vs_pure_nmse"] < 1e-6, case
                assert case["triton_mixed_vs_native_mixed_nmse"] < 1e-6, case
                if length == 54:
                    assert case["triton_pure_vs_pure_nmse"] == 0, case
                    assert case["triton_mixed_vs_native_mixed_nmse"] == 0, case
            if name == "main_9482":
                main_inputs = q, k, v
                main_packet = packet, v8, s8, route

        q, k, v = main_inputs
        packet, v8, s8, route = main_packet
        k_work = torch.empty_like(k)

        def online():
            return sageattn3_blackwell(q, k_work, v,
                                       is_causal=False, per_block_mean=True)

        report["timings"]["pure_core"] = time_cuda(
            lambda: fp4attn_cuda.fwd(*packet))
        report["timings"]["mixed_core"] = time_cuda(
            lambda: mixed_cuda.fwd_mixed(*packet, v8, s8, route))
        report["timings"]["pure_online"] = time_cuda(
            online, prepare=lambda: k_work.copy_(k))
        for key, value in report["timings"].items():
            print(key, value["mean_ms"], "ms, CV", value["cv_percent"],
                  "%", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
