#!/usr/bin/env python3
"""Recompute E122 attribution, Amdahl checks, and length-sweep repeatability."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "results/research/E122"
ARMS = ("bf16_sdpa", "sage2_default", "sage3_fixed", "split_mixed")


def mean(values):
    return statistics.mean(values)


def main() -> None:
    profile = json.loads((FOLDER / "h3_forward_profile_refactored.json").read_text())
    runs = [json.loads((FOLDER / name).read_text()) for name in
            ("mixed_length_sweep.json", "mixed_length_sweep_repeat2.json",
             "mixed_length_sweep_refactored.json")]
    assert set(profile["arms"]) == set(ARMS)
    assert profile["model_load_excluded"] is True and profile["steps"] == [0, 10]
    assert profile["linears"]["count"] == 200
    assert all(x["status"] == "complete" and len(x["route_boundary_checks"]) == 8
               and all(t["status"] == "passed" for t in x["route_boundary_checks"])
               for x in runs)
    lengths = runs[0]["lengths"]
    assert all(run["lengths"] == lengths for run in runs)
    assert all([r["tokens"] for r in run["rows"]] == lengths for run in runs)

    arm_rows = {}
    for arm in ARMS:
        rows = profile["arms"][arm]
        assert [x["step"] for x in rows] == [0, 10]
        b = [x["profiled"]["breakdown"] for x in rows]
        arm_rows[arm] = {
            "direct_dit_seconds_mean": mean(x["direct"]["wall_seconds"] for x in rows),
            "profiled_dit_seconds_mean": mean(x["profiled"]["wall_seconds"] for x in rows),
            "main_attention_seconds_mean": mean(
                x["attention_helper"]["main"]["ms"] / 1000 for x in b),
            "refiner_attention_seconds_mean": mean(
                x["attention_helper"]["refiner"]["ms"] / 1000 for x in b),
            "linear_module_seconds_mean": mean(
                x["all_linear_module_ms"] / 1000 for x in b),
            "linear_gemm_seconds_mean": mean(
                x["linear_gemm_ms"] / 1000 for x in b),
            "linear_nongemm_estimate_seconds_mean": mean(
                x["linear_non_gemm_estimate_ms"] / 1000 for x in b),
            "other_seconds_mean": mean(
                x["other_exclusive_estimate_ms"] / 1000 for x in b),
            "nmse_vs_true_bf16": {name: mean(
                x["vs_true_bf16_dit_nmse"][name] for x in rows)
                for name in ("video", "audio")},
            "linear_categories_seconds": {name: mean(
                x["linear_module_including_weight_load"][name]["ms"] / 1000
                for x in b) for name in b[0]["linear_module_including_weight_load"]},
        }
        assert all(math.isfinite(v) and v >= 0 for v in arm_rows[arm].values()
                   if isinstance(v, (int, float)))
    bf16 = arm_rows["bf16_sdpa"]
    for arm in ARMS:
        row = arm_rows[arm]
        row["observed_speedup_vs_true_bf16"] = (
            bf16["direct_dit_seconds_mean"] / row["direct_dit_seconds_mean"])
        row["main_attention_speedup_vs_true_bf16"] = (
            bf16["main_attention_seconds_mean"] /
            row["main_attention_seconds_mean"])
        # Hold the non-attention part of the true BF16 run fixed.
        row["amdahl_predicted_dit_speedup"] = (
            bf16["profiled_dit_seconds_mean"] /
            (bf16["profiled_dit_seconds_mean"] -
             bf16["main_attention_seconds_mean"] +
             row["main_attention_seconds_mean"]))

    sweep_rows = []
    for index, length in enumerate(lengths):
        pair = [x["rows"][index] for x in runs]
        assert all(x["tokens"] == length and x["heads"] == 56 and
                   x["consistency_gate"] == "passed" for x in pair)
        assert all(x["selected_tiles"] == pair[0]["selected_tiles"] and
                   x["selected_after_tile_127"] == pair[0]["selected_after_tile_127"]
                   for x in pair)
        timings = {arm: [x["timing"][arm]["mean_ms"] for x in pair]
                   for arm in ("bf16_sdpa", "sage3_fixed", "native_mixed", "split_mixed")}
        row = {"tokens": length, "selected_tiles": pair[0]["selected_tiles"],
               "selected_after_tile_127": pair[0]["selected_after_tile_127"],
               "timing_ms_mean_across_processes": {arm: mean(v)
                                                    for arm, v in timings.items()},
               "timing_ms_by_process": timings,
               "max_split_vs_native_nmse": max(x["split_vs_native_nmse"] for x in pair),
               "nmse_vs_true_bf16": pair[0]["nmse_vs_bf16"]}
        times = row["timing_ms_mean_across_processes"]
        row["sage3_speedup_vs_bf16"] = times["bf16_sdpa"] / times["sage3_fixed"]
        row["split_speedup_vs_bf16"] = times["bf16_sdpa"] / times["split_mixed"]
        row["split_slowdown_vs_sage3"] = times["split_mixed"] / times["sage3_fixed"]
        sweep_rows.append(row)
    summary = {"format": "e122-h3-attribution-summary-v1",
               "gpu": profile["gpu"], "steps": profile["steps"],
               "model_load_excluded": True, "profile_arms": arm_rows,
               "length_sweep_independent_processes": len(runs),
               "length_sweep_benchmark_calls_per_process": 32,
               "length_sweep": sweep_rows,
               "max_split_vs_native_nmse_all_lengths": max(
                   x["max_split_vs_native_nmse"] for x in sweep_rows),
               "tail_route_at_full_length": sweep_rows[-1]["selected_after_tile_127"],
               "status": "complete"}
    path = FOLDER / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(path),
                      "true_bf16_to_sage3_dit_speedup": arm_rows["sage3_fixed"]["observed_speedup_vs_true_bf16"],
                      "true_bf16_to_split_dit_speedup": arm_rows["split_mixed"]["observed_speedup_vs_true_bf16"],
                      "max_split_native_nmse": summary["max_split_vs_native_nmse_all_lengths"]}))


if __name__ == "__main__":
    main()
