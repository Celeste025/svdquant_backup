#!/usr/bin/env python3
"""Validate and time corrected Sage3/mixed attention across H3 token lengths."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import triton

import e122_h3_helpers as exp
from h3_mixed_sage3_api import (_compact_route_all, native_mixed_attention,
                                split_mixed_attention)
from sageattn3 import sageattn3_blackwell
from sageattn3.api import preprocess_qkv
import sage3_mixed_router_sparse_cuda

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/"
               "E121_h3_split_generation/race/block25_qkv_h56.pt")
OUT = exp.ROOT / "results/research/E122/mixed_length_sweep.json"
LENGTHS = (127, 128, 129, 255, 256, 1024, 4096, 8192,
           16384, 16385, 22677)


def nmse(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    return float((x - y).square().sum() / y.square().sum())


def cuda_benchmark(fn, *, warmups=4, batches=4, repeats=8) -> dict:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    batch_means = []
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
        batch_means.append(statistics.mean(s.elapsed_time(e) for s, e in events))
    return {"mean_ms": statistics.mean(batch_means),
            "batch_means_ms": batch_means, "calls": batches * repeats,
            "warmups": warmups}


def test_route_boundaries(device: str) -> list[dict]:
    checks = []
    for tiles in (1, 2, 127, 128, 129, 178, 256, 257):
        route = torch.zeros((1, 3, tiles), dtype=torch.int32, device=device)
        route[:, :, 0] = 1
        route[:, :, -1] = 1
        ids = torch.empty_like(route)
        counts = torch.empty((1, 3), dtype=torch.int32, device=device)
        _compact_route_all[(3,)](route, ids, counts, tiles,
                                  triton.next_power_of_2(tiles))
        expected = 1 if tiles == 1 else 2
        assert bool(torch.all(counts == expected))
        assert bool(torch.all(ids[:, :, 0] == 0))
        if tiles > 1:
            assert bool(torch.all(ids[:, :, 1] == tiles - 1))
        checks.append({"tiles": tiles, "selected": expected, "status": "passed"})
    return checks


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--output", type=Path, default=OUT)
    opts = parser.parse_args()
    exp.check_backend()
    assert opts.capture.is_file(), f"missing full-head capture: {opts.capture}"
    payload = torch.load(opts.capture, map_location="cpu", weights_only=True,
                         mmap=True)
    assert set(payload) == {"q", "k", "v"}
    assert payload["q"].shape == payload["k"].shape == payload["v"].shape
    assert payload["q"].ndim == 4 and payload["q"].shape[0] == 1
    assert payload["q"].shape[1] == 56 and payload["q"].shape[3] == 128
    assert payload["q"].shape[2] >= max(LENGTHS)
    report = {"format": "e122-mixed-length-sweep-v1",
              "capture": str(opts.capture), "gpu": torch.cuda.get_device_name(),
              "head_count": 56, "head_dim": 128, "causal": False,
              "route_boundary_checks": test_route_boundaries("cuda"),
              "lengths": list(LENGTHS), "rows": [], "status": "running"}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    exp.save_json(opts.output, report)
    for length in LENGTHS:
        q, k, v = (payload[name][:, :, :length].to("cuda").contiguous()
                   for name in ("q", "k", "v"))
        fns = {
            "bf16_sdpa": lambda: F.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=128 ** -0.5),
            "sage3_fixed": lambda: sageattn3_blackwell(
                q, k.clone(), v, is_causal=False, per_block_mean=True),
            "native_mixed": lambda: native_mixed_attention(
                q, k, v, threshold=0.10),
            "split_mixed": lambda: split_mixed_attention(
                q, k, v, threshold=0.10),
        }
        outputs = {name: fn() for name, fn in fns.items()}
        torch.cuda.synchronize()
        assert all(x.shape == q.shape and x.dtype == torch.bfloat16 and
                   bool(torch.isfinite(x).all()) for x in outputs.values())
        qp, kp, vp, _ = preprocess_qkv(q, k.clone(), v, True)
        _, _, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(vp, 0.10)
        tiles = triton.cdiv(length, 128)
        route_total = int(route.sum())
        route_tail = int(route[:, :, 128:].sum()) if tiles > 128 else 0
        row = {"tokens": length, "heads": 56, "tiles_per_head": tiles,
               "selected_tiles": route_total, "selected_after_tile_127": route_tail,
               "total_tiles": route.numel(),
               "nmse_vs_bf16": {name: nmse(x, outputs["bf16_sdpa"])
                                for name, x in outputs.items() if name != "bf16_sdpa"},
               "split_vs_native_nmse": nmse(outputs["split_mixed"],
                                            outputs["native_mixed"]),
               "split_vs_sage3_nmse": nmse(outputs["split_mixed"],
                                           outputs["sage3_fixed"]),
               "timing": {name: cuda_benchmark(fn) for name, fn in fns.items()}}
        row["speedup_vs_bf16"] = {
            name: row["timing"]["bf16_sdpa"]["mean_ms"] / timed["mean_ms"]
            for name, timed in row["timing"].items() if name != "bf16_sdpa"}
        row["split_slowdown_vs_sage3"] = (
            row["timing"]["split_mixed"]["mean_ms"] /
            row["timing"]["sage3_fixed"]["mean_ms"])
        assert math.isfinite(row["split_vs_native_nmse"])
        row["consistency_gate"] = (
            "passed" if row["split_vs_native_nmse"] < 1e-4 else "failed")
        report["rows"].append(row)
        exp.save_json(opts.output, report)
        print(f"L={length} routes={route_total}/{route.numel()} tail={route_tail} "
              f"split/native_nmse={row['split_vs_native_nmse']:.3g} "
              f"BF16={row['timing']['bf16_sdpa']['mean_ms']:.3f}ms "
              f"Sage3={row['timing']['sage3_fixed']['mean_ms']:.3f}ms "
              f"split={row['timing']['split_mixed']['mean_ms']:.3f}ms",
              flush=True)
        del q, k, v, outputs, qp, kp, vp, route
    report["status"] = ("complete" if all(x["consistency_gate"] == "passed"
                                          for x in report["rows"]) else "failed_gate")
    exp.save_json(opts.output, report)
    if report["status"] != "complete":
        raise RuntimeError("one or more length consistency gates failed")
    print(json.dumps({"status": report["status"], "output": str(opts.output)}),
          flush=True)


if __name__ == "__main__":
    main()
