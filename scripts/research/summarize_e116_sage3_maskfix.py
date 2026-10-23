#!/usr/bin/env python3
"""Validate and summarize three same-GPU Sage3 mask-fix A/B runs."""
from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research/E116"


def read(name: str) -> dict:
    path = OUT / f"{name}_gpu3.json"
    value = json.loads(path.read_text())
    assert value["format"] == "e116-sage3-maskfix-ab-v1"
    assert "RTX PRO 5000" in value["gpu"]
    assert set(value["cases"]) == {"control_32", "pad_54", "main_9482"}
    return value


def main() -> None:
    baseline = [read(name) for name in ("baseline", "baseline2", "baseline3")]
    fixed = [read(name) for name in ("fixed", "fixed2", "fixed3")]
    assert len({x["capture"] for x in baseline + fixed}) == 1
    assert len({x["pure_extension"] for x in baseline}) == 1
    assert len({x["pure_extension"] for x in fixed}) == 1
    for b, f in zip(baseline, fixed):
        assert b["cases"]["control_32"]["triton_pure_vs_pure_nmse"] == 0
        assert f["cases"]["control_32"]["triton_pure_vs_pure_nmse"] == 0
        assert b["cases"]["pad_54"]["triton_pure_vs_pure_nmse"] > 0.006
        assert f["cases"]["pad_54"]["triton_pure_vs_pure_nmse"] == 0
        assert f["cases"]["pad_54"]["triton_mixed_vs_native_mixed_nmse"] == 0
        assert b["cases"]["main_9482"]["triton_pure_vs_pure_nmse"] > 1e-5
        assert f["cases"]["main_9482"]["triton_pure_vs_pure_nmse"] < 1e-7
        assert f["cases"]["main_9482"]["triton_mixed_vs_native_mixed_nmse"] < 1e-7
        assert all(v["cv_percent"] < 2 for v in b["timings"].values())
        assert all(v["cv_percent"] < 2 for v in f["timings"].values())

    timings = {}
    for scope in ("pure_core", "pure_online", "mixed_core"):
        old = [x["timings"][scope]["mean_ms"] for x in baseline]
        new = [x["timings"][scope]["mean_ms"] for x in fixed]
        timings[scope] = {
            "baseline_runs_ms": old,
            "fixed_runs_ms": new,
            "baseline_mean_ms": statistics.mean(old),
            "fixed_mean_ms": statistics.mean(new),
            "fixed_vs_baseline_latency_percent":
                (statistics.mean(new) / statistics.mean(old) - 1) * 100,
            "paired_latency_percent": [(y / x - 1) * 100
                                       for x, y in zip(old, new)],
        }
    report = {
        "format": "e116-sage3-maskfix-summary-v1",
        "capture": baseline[0]["capture"],
        "gpu": baseline[0]["gpu"],
        "repeat_count_per_arm": 3,
        "cuda_event_calls_per_scope_per_run": 96,
        "timings": timings,
        "baseline_54_nmse_vs_logical_triton": baseline[0]["cases"]["pad_54"]["triton_pure_vs_pure_nmse"],
        "fixed_54_nmse_vs_logical_triton": fixed[0]["cases"]["pad_54"]["triton_pure_vs_pure_nmse"],
        "baseline_main_nmse_vs_logical_triton": baseline[0]["cases"]["main_9482"]["triton_pure_vs_pure_nmse"],
        "fixed_main_nmse_vs_logical_triton": fixed[0]["cases"]["main_9482"]["triton_pure_vs_pure_nmse"],
        "fixed_main_mixed_nmse_vs_logical_triton": fixed[0]["cases"]["main_9482"]["triton_mixed_vs_native_mixed_nmse"],
        "status": "complete",
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
