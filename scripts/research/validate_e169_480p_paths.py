#!/usr/bin/env python3
"""Validate and time dense-INT8 route capture plus later-step route reuse."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import torch
from torch.profiler import profile, ProfilerActivity


ROOT = Path(__file__).resolve().parents[2]
ANEMOI = ROOT / "third_party/anemoi-code"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E169_h3_480p5s_attention")
OUT = ROOT / "results/research/E169"
sys.path.insert(0, str(ANEMOI))

from anemoi import (  # noqa: E402
    AttentionRouteCache,
    NVFP4Calibration,
    QuantConfig,
    SparseConfig,
    VisualLayout,
    anemoi_attention,
)
import anemoi.layers.attention.mpa.executor as executor  # noqa: E402


def _nmse(output: torch.Tensor, reference: torch.Tensor) -> float:
    error_energy = 0.0
    reference_energy = 0.0
    for begin in range(0, output.size(1), 512):
        output_chunk = output[:, begin : begin + 512].float()
        reference_chunk = reference[:, begin : begin + 512].float()
        error_energy += float(
            (output_chunk - reference_chunk).square().sum(dtype=torch.float64)
        )
        reference_energy += float(reference_chunk.square().sum(dtype=torch.float64))
    return error_energy / reference_energy


@torch.inference_mode()
def _measure(function, *, warmup: int = 3, repeats: int = 8) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        output = function()
        assert torch.isfinite(output).all()
        del output
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = function()
        end.record()
        end.synchronize()
        assert torch.isfinite(output).all()
        values.append(begin.elapsed_time(end))
        del output
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "samples_ms": values,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, choices=(124,), default=124)
    parser.add_argument("--query-block", type=int, choices=(64, 128), default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--skip-fraction", type=float, default=0.75)
    parser.add_argument("--nvfp4-fraction", type=float, default=0.6,
                        help="NVFP4 fraction among retained tiles; remainder is INT8")
    parser.add_argument("--output", type=Path, help="Explicit report path for alternate policies")
    parser.add_argument("--kernel-profile", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.skip_fraction <= 1 or not 0 <= args.nvfp4_fraction <= 1:
        parser.error("fractions must be in [0, 1]")
    if (args.skip_fraction != 0.75 or args.nvfp4_fraction != 0.6) and args.output is None:
        parser.error("alternate policies require --output to preserve baseline reports")
    torch.set_num_threads(6)
    device = torch.device("cuda:0")
    source = DATA / f"race_480p_{args.frames}f_step0_block0.pt"
    payload = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    q_hnd, k_hnd, v_hnd = [
        payload[name].to(device).contiguous() for name in ("q", "k", "v")
    ]
    q, k, v = [tensor.transpose(1, 2).contiguous() for tensor in (q_hnd, k_hnd, v_hnd)]
    prefix_tokens = int(payload["prefix_tokens"])
    layer = int(payload["layer"])
    layout = VisualLayout(tuple(payload["video_grid"]), prefix_tokens)
    scale_payload = json.loads(
        (ANEMOI / "anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json").read_text()
    )["scales"]
    calibration = NVFP4Calibration(
        *(float(scale_payload[key][layer]) for key in (
            "q_tensor_scale",
            "k_tensor_scale",
            "v_tensor_scale",
        ))
    )
    sparse = SparseConfig(
        query_block_size=args.query_block,
        sparsity_ratio=args.skip_fraction,
        maxpool_weight=0.2,
    )
    mixed = QuantConfig(
        nvfp4_ratio=args.nvfp4_fraction,
        int8_ratio=1.0 - args.nvfp4_fraction,
        fp16_ratio=0.0,
        prefix_kv_precision="int8",
        prefix_query_precision="int8",
    )
    dense = QuantConfig(
        nvfp4_ratio=0.0,
        int8_ratio=1.0,
        fp16_ratio=0.0,
        prefix_kv_precision="int8",
        prefix_query_precision="int8",
    )
    cache = AttentionRouteCache()

    def legacy_mixed() -> torch.Tensor:
        return anemoi_attention(
            q,
            k,
            v,
            layout=layout,
            layer=layer,
            sparse_config=sparse,
            quant_config=mixed,
            calibration=calibration,
        )

    def first_step() -> torch.Tensor:
        cache.clear()
        return anemoi_attention(
            q,
            k,
            v,
            layout=layout,
            layer=layer,
            sparse_config=sparse,
            quant_config=mixed,
            calibration=calibration,
            first_step=True,
            route_cache=cache,
        )

    def reuse_step() -> torch.Tensor:
        return anemoi_attention(
            q,
            k,
            v,
            layout=layout,
            layer=layer,
            sparse_config=sparse,
            quant_config=mixed,
            calibration=calibration,
            first_step=False,
            route_cache=cache,
        )

    def dense_int8() -> torch.Tensor:
        return anemoi_attention(
            q,
            k,
            v,
            layout=layout,
            layer=layer,
            sparse_config=SparseConfig(
                query_block_size=args.query_block,
                sparsity_ratio=0.0,
                maxpool_weight=0.0,
            ),
            quant_config=dense,
        )

    first_output = first_step()
    assert len(cache) == 1
    assert cache.allocated_bytes > 0
    with (
        mock.patch.object(
            executor,
            "sm120_h3_draft_probability",
            side_effect=AssertionError("reuse called DraftMap"),
        ),
        mock.patch.object(
            executor,
            "_route_sm120_probability",
            side_effect=AssertionError("reuse reran route selection"),
        ),
    ):
        reuse_output = reuse_step()
    dense_output = dense_int8()
    assert first_output.shape == reuse_output.shape == dense_output.shape
    for name, output in {
        "first": first_output,
        "reuse": reuse_output,
        "dense": dense_output,
    }.items():
        assert torch.isfinite(output).all(), name
    first_vs_dense_nmse = _nmse(first_output, dense_output)
    reuse_vs_dense_nmse = _nmse(reuse_output, dense_output)
    del first_output, dense_output
    torch.cuda.empty_cache()
    legacy_output = legacy_mixed()
    assert legacy_output.shape == reuse_output.shape
    assert torch.isfinite(legacy_output).all()
    reuse_vs_legacy_nmse = _nmse(reuse_output, legacy_output)
    del legacy_output, reuse_output
    torch.cuda.empty_cache()

    timings = {
        "dense_int8_no_capture": _measure(
            dense_int8, warmup=args.warmup, repeats=args.repeats
        ),
        "dense_int8_first_step_capture": _measure(
            first_step, warmup=args.warmup, repeats=args.repeats
        ),
        "cached_exact_route_reuse": _measure(
            reuse_step, warmup=args.warmup, repeats=args.repeats
        ),
        "legacy_proxy_route_each_step": _measure(
            legacy_mixed, warmup=args.warmup, repeats=args.repeats
        ),
    }
    kernel_profiles = {}
    if args.kernel_profile:
        for name, function in (
            ("dense_int8_no_capture", dense_int8),
            ("dense_int8_first_step_capture", first_step),
            ("cached_exact_route_reuse", reuse_step),
            ("legacy_proxy_route_each_step", legacy_mixed),
        ):
            function()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                output = function()
                torch.cuda.synchronize()
                del output
            kernel_profiles[name] = [
                {"name": event.name,
                 "device_type": str(event.device_type),
                 "self_device_time_us": float(event.self_device_time_total)}
                for event in prof.events()
                if "cuda" in str(event.device_type).lower()
            ]
    report = {
        "format": "e154-exact-route-reuse-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "capture": str(source),
        "shape_bshd": list(q.shape),
        "video_grid": list(payload["video_grid"]),
        "prefix_tokens": prefix_tokens,
        "layer": layer,
        "policy": {
            "query_block": args.query_block,
            "skip_fraction": args.skip_fraction,
            "retained_nvfp4_fraction": args.nvfp4_fraction,
            "retained_int8_fraction": 1.0 - args.nvfp4_fraction,
            "first_step": (
                "dense INT8; actual quantized QK physical-tile maximum captured "
                "in kernel (adjacent K64 tiles merged for Q128 routing)"
            ),
            "later_steps": "reuse cached first-step route; no DraftMap pooling/scoring/sorting",
        },
        "checks": {
            "cache_entries": len(cache),
            "cache_bytes_one_layer": cache.allocated_bytes,
            "reuse_draftmap_and_route_selection_mocked_to_raise": True,
            "first_vs_dense_int8_nmse": first_vs_dense_nmse,
            "reuse_vs_legacy_proxy_route_nmse": reuse_vs_legacy_nmse,
            "reuse_vs_dense_int8_nmse": reuse_vs_dense_nmse,
        },
        "timings": timings,
        "kernel_profiles": kernel_profiles,
        "status": "complete",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = args.output or OUT / (
        f"race_768p_{args.frames}f_step0_block0_q{args.query_block}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
