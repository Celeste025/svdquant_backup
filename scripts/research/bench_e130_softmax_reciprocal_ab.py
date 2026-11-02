#!/usr/bin/env python3
"""Deprecated same-process A/B attempt; use the separate-process harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/sage3_h3_maskfix"))
from sageattn3 import sageattn3_blackwell
import sageattn3.api as api

BASELINE_SO = Path("/tmp/e130-softmax-baseline/fp4attn_cuda.cpython-312-x86_64-linux-gnu.so")
CAPTURES = {
    "short": Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt"),
    "long": Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt"),
}
OUTPUT = ROOT / "results/research/E130/softmax_reciprocal_ab_gpu2.json"


def load_baseline():
    spec = importlib.util.spec_from_file_location("fp4attn_cuda", BASELINE_SO)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load baseline extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              modules: dict[str, object]) -> dict:
    k_work = torch.empty_like(k)
    outputs = {}
    timings = {name: [] for name in modules}

    def invoke(name: str) -> torch.Tensor:
        api.fp4attn_cuda = modules[name]
        k_work.copy_(k)
        return sageattn3_blackwell(q, k_work, v, is_causal=False, per_block_mean=True)

    with torch.inference_mode():
        for name in modules:
            outputs[name] = invoke(name).clone()
        difference = (outputs["reciprocal"].float() - outputs["baseline"].float())
        energy = outputs["baseline"].float().square().sum()
        nmse = (difference.square().sum() / energy).item()
        maximum = difference.abs().max().item()
        for order in (tuple(modules), tuple(reversed(modules)), tuple(modules)):
            for name in order:
                api.fp4attn_cuda = modules[name]
                for _ in range(25):
                    k_work.copy_(k)
                    sageattn3_blackwell(q, k_work, v, is_causal=False, per_block_mean=True)
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(48)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(48)]
                for start, end in zip(starts, ends):
                    k_work.copy_(k)
                    start.record()
                    sageattn3_blackwell(q, k_work, v, is_causal=False, per_block_mean=True)
                    end.record()
                torch.cuda.synchronize()
                timings[name].append(statistics.mean(s.elapsed_time_ms(e) for s, e in zip(starts, ends)))
    means = {name: statistics.mean(values) for name, values in timings.items()}
    return {"shape": list(q.shape), "rounds_ms": timings, "means_ms": means,
            "speedup": means["baseline"] / means["reciprocal"],
            "nmse_vs_baseline": nmse, "max_abs_diff_vs_baseline": maximum,
            "finite": all(bool(torch.isfinite(x).all()) for x in outputs.values())}


def main() -> None:
    raise SystemExit("Use bench_e130_softmax_reciprocal_single.py in separate processes; Python caches extension modules by name")
    torch.cuda.set_device(0)
    baseline = load_baseline()
    reciprocal = api.fp4attn_cuda
    if Path(baseline.__file__).resolve() == Path(reciprocal.__file__).resolve():
        raise RuntimeError("A/B modules point to the same binary")
    modules = {"baseline": baseline, "reciprocal": reciprocal}
    results = {}
    for name, path in CAPTURES.items():
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if name == "short":
            q, k, v = [payload[key].to("cuda:0").transpose(0, 1).unsqueeze(0)
                       for key in ("q", "k", "v")]
        else:
            q, k, v = [payload[key].to("cuda:0") for key in ("q", "k", "v")]
        results[name] = one_shape(q, k, v, modules)
        print(name, json.dumps(results[name]), flush=True)
        del q, k, v, payload
    api.fp4attn_cuda = reciprocal
    report = {"format": "e130-softmax-reciprocal-ab-v1", "gpu": torch.cuda.get_device_name(),
              "baseline_so": str(BASELINE_SO), "reciprocal_so": reciprocal.__file__,
              "scope": "full Sage3 API; K restoration outside CUDA-event interval",
              "results": results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
