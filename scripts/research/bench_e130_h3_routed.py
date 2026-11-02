#!/usr/bin/env python3
"""Measure the H3 NHD-to-NHD Sage3 route, including layout and final copy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research"),
                str(ROOT / "third_party/sage3_h3_maskfix")]
from bench_h3_sage_real_gpu0 import metrics
from bench_h3_sage_two_scope import measure
from e130_sage3_preprocess import attention

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt")
OUTPUT = ROOT / "results/research/E130/h3_routed_sage3_gpu2.json"
VARIANTS = ("current_clone", "official_no_clone", "api_fast_no_clone",
            "api_fast_no_crop", "fused_with_crop",
            "fused_no_crop", "fused_exact_no_crop", "fused_k_exact_no_crop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sage3-dir", type=Path,
                        default=ROOT / "third_party/sage3_h3_maskfix")
    parser.add_argument("--delta-dtype", choices=("fp32", "bf16"), default="fp32")
    opts = parser.parse_args()
    sys.path.insert(0, str(opts.sage3_dir.resolve()))
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    n = 9482
    q, k, v = [payload[name][:n].to("cuda:0") for name in ("q", "k", "v")]
    if any(x.shape != (n, 56, 128) for x in (q, k, v)):
        raise RuntimeError("unexpected H3 NHD shape")

    def call(name: str):
        qh, kh, vh = (x.transpose(0, 1).unsqueeze(0).contiguous()
                      for x in (q, k, v))
        if name in ("current_clone", "official_no_clone", "api_fast_no_clone",
                    "api_fast_no_crop"):
            result = sageattn3_blackwell(
                qh, kh.clone() if name == "current_clone" else kh, vh,
                is_causal=False, per_block_mean=True,
                fast_preprocess=name.startswith("api_fast"),
                fast_contiguous_output=name != "api_fast_no_crop",
            )
        else:
            result = attention(qh, kh, vh, api,
                               fuse_q_pad_mean=True, implicit_kv_pad=True,
                               exact_delta=name in ("fused_exact_no_crop",
                                                    "fused_k_exact_no_crop"),
                               fuse_k_center_pad=name == "fused_k_exact_no_crop",
                               delta_dtype=(torch.bfloat16 if opts.delta_dtype == "bf16"
                                            else torch.float32),
                               contiguous_output=name == "fused_with_crop")
        out = torch.empty_like(q)
        out.copy_(result[0].transpose(0, 1))
        return out

    with torch.inference_mode():
        reference = call("current_clone")
        checks = {name: metrics(call(name), reference) for name in VARIANTS}
        if any(result["nmse"] > 1e-7 for result in checks.values()):
            raise RuntimeError(f"routed output mismatch: {checks}")
        timings = {name: [] for name in VARIANTS}
        for ordering in (VARIANTS, tuple(reversed(VARIANTS))):
            for name in ordering:
                row = measure(lambda name=name: call(name))
                timings[name].append(row)
                print(name, row["mean_ms"], "ms", flush=True)
    means = {name: statistics.mean(row["mean_ms"] for row in entries)
             for name, entries in timings.items()}
    report = {"format": "e130-h3-routed-sage3-ab-v1",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "capture": str(CAPTURE), "shape_nhd": list(q.shape),
              "gpu": torch.cuda.get_device_name(),
              "sage3_source": str(opts.sage3_dir.resolve()),
              "delta_dtype": opts.delta_dtype,
              "scope": "H3 QKV NHD -> HND contiguous layout, Sage3 preprocess/core, NHD output copy; no QKV projection or 54-token padding-tail attention",
              "checks_vs_current_clone": checks, "means_ms": means,
              "speedup_vs_current_clone": {name: means["current_clone"] / value
                                           for name, value in means.items()},
              "rounds": timings}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"means_ms": means, "speedup_vs_current_clone": report["speedup_vs_current_clone"]}, indent=2))


if __name__ == "__main__":
    main()
