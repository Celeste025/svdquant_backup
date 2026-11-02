#!/usr/bin/env python3
"""Pair dense INT8 attention with and without in-kernel QK importance capture."""

from __future__ import annotations

import json
import math
import statistics
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
ANEMOI = ROOT / "third_party/anemoi-code"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E151_long_h3_attention")
OUT = ROOT / "results/research/E154/capture_overhead.json"
sys.path.insert(0, str(ANEMOI))

from anemoi.layers.attention.mpa.backends.sm120_q64 import (  # noqa: E402
    sm120_q64_dense_int8_importance_attention,
    sm120_q64_prefix_int8_attention,
)
from anemoi.layers.attention.mpa.backends.sm120_q128 import (  # noqa: E402
    sm120_q128_dense_int8_importance_attention,
    sm120_q128_prefix_int8_attention,
)
from anemoi.layers.attention.mpa.executor import (  # noqa: E402
    _expand_q128_valid_k,
    _prepare_h3_sm120_inputs,
)
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout  # noqa: E402


def _measure(function, warmup: int, repeats: int) -> dict[str, object]:
    for _ in range(warmup):
        result = function()
        del result
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = function()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
        del result
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _nmse_chunked(output: torch.Tensor, reference: torch.Tensor) -> float:
    error = 0.0
    energy = 0.0
    for begin in range(0, output.size(2), 512):
        a = output[:, :, begin : begin + 512].float()
        b = reference[:, :, begin : begin + 512].float()
        error += float((a - b).square().sum(dtype=torch.float64))
        energy += float(b.square().sum(dtype=torch.float64))
    return error / energy


@torch.inference_mode()
def run_case(frames: int, query_block: int, *, quick: bool = False) -> dict[str, object]:
    payload = torch.load(
        DATA / f"race_768p_{frames}f_step0_block0.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    q, k, v = [
        payload[name].cuda().transpose(1, 2).contiguous()
        for name in ("q", "k", "v")
    ]
    prefix_tokens = int(payload["prefix_tokens"])
    video_shape = tuple(payload["video_grid"])
    layout = materialize_ragged_2d_layout(
        q.device,
        frames=video_shape[0],
        height=video_shape[1],
        width=video_shape[2],
        logical_block=query_block,
        enable_anchors=False,
    )
    video_counts = layout.counts.view(1, -1).contiguous()
    prepared = _prepare_h3_sm120_inputs(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        video_token_indices=layout.indices,
        video_slot_valid=layout.slot_valid,
        video_valid_counts=layout.counts,
        prefix_tokens=prefix_tokens,
        query_block_size=query_block,
        phases=("int8",),
        has_prefix_query_int8=True,
        has_pool=False,
        has_maxpool=False,
        global_scales=None,
    )
    int8_operands = prepared[7]
    prefix_blocks = math.ceil(prefix_tokens / 64)
    prefix_counts = (
        torch.clamp(
            prefix_tokens
            - torch.arange(prefix_blocks, device=q.device, dtype=torch.int32) * 64,
            min=0,
            max=64,
        )
        .view(1, -1)
        .contiguous()
    )
    valid_k = (
        _expand_q128_valid_k(prefix_counts, video_counts)
        if query_block == 128
        else torch.cat((prefix_counts, video_counts), dim=1).contiguous()
    )
    prefix_function = (
        sm120_q64_prefix_int8_attention
        if query_block == 64
        else sm120_q128_prefix_int8_attention
    )
    capture_function = (
        sm120_q64_dense_int8_importance_attention
        if query_block == 64
        else sm120_q128_dense_int8_importance_attention
    )
    q8, q8_scale = int8_operands[:2]

    def dense_only() -> torch.Tensor:
        return prefix_function(
            q8, q8_scale, int8_operands, valid_k, q8.size(2)
        )

    def dense_capture() -> tuple[torch.Tensor, torch.Tensor]:
        return capture_function(int8_operands, valid_k)

    plain = dense_only()
    captured, importance = dense_capture()
    result = {
        "frames": frames,
        "query_block": query_block,
        "tokens": int(q.size(1)),
        "prepared_query_tokens": int(q8.size(2)),
        "importance_shape": list(importance.shape),
        "capture_vs_plain_nmse": _nmse_chunked(captured, plain),
    }
    del plain, captured, importance
    torch.cuda.empty_cache()
    warmup, repeats = ((3, 10) if quick else ((10, 30) if frames == 124 else (3, 10)))
    timings = {}
    for order in (("plain", "capture"), ("capture", "plain")):
        for name in order:
            function = dense_only if name == "plain" else dense_capture
            timings.setdefault(name, []).append(_measure(function, warmup, repeats))
    for name, rounds in timings.items():
        result[f"{name}_mean_ms"] = statistics.mean(
            float(round_["mean_ms"]) for round_ in rounds
        )
        result[f"{name}_rounds"] = rounds
    result["capture_overhead_percent"] = 100.0 * (
        result["capture_mean_ms"] / result["plain_mean_ms"] - 1.0
    )
    del q, k, v, prepared, int8_operands, valid_k
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the two representative 38k-token cases with fewer repeats.",
    )
    args = parser.parse_args()
    torch.set_num_threads(6)
    report = {
        "format": "e154-dense-int8-capture-overhead-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "scope": "prepacked native dense INT8 attention kernel only; CUDA events",
        "cases": [],
    }
    cases = ((124, 64), (124, 128)) if args.quick else ((124, 64), (124, 128), (362, 64))
    for frames, query_block in cases:
        row = run_case(frames, query_block, quick=args.quick)
        report["cases"].append(row)
        print(json.dumps(row, indent=2), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
