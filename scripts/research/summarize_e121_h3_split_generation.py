#!/usr/bin/env python3
"""Validate and summarize the paired E121 H3 generation reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import torch

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_ARMS = ("bf16", "sage3_fixed", "split_mixed")
MEASURE_ARMS = (*PREFLIGHT_ARMS, "bf16_warm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="race", choices=("race", "clap0"))
    args = parser.parse_args()
    folder = ROOT / "results/research/E121" / args.case
    preflight = json.loads((folder / "attention_preflight_fixed.json").read_text())
    assert (preflight["status"] == "complete" and
            set(preflight["arms"]) == set(PREFLIGHT_ARMS))
    rows = {}
    latents = {}
    common_settings = None
    initial_hashes = None
    for arm in MEASURE_ARMS:
        d = json.loads((folder / f"{arm}_denoise.json").read_text())
        v = json.loads((folder / f"{arm}_video.json").read_text())
        assert d["status"] == "denoise_complete" and len(d["steps"]) == 20
        assert d["arm"] == v["arm"] == arm and d["case"] == v["case"] == args.case
        assert [x["step"] for x in d["steps"]] == list(range(20))
        assert all(x["attention_main_calls"] == 50 and
                   x["attention_refiner_calls"] == 2 and
                   x["main_tokens"] == d["steps"][0]["main_tokens"] and
                   math.isfinite(x["dit_seconds"]) and x["dit_seconds"] > 0
                   for x in d["steps"])
        assert v["frames"] == 124 and v["width"] == 1024 and v["height"] == 576
        assert Path(v["video"]).is_file()
        if common_settings is None:
            common_settings = d["settings"]
            initial_hashes = d["initial_latent_sha256"]
        assert d["settings"] == common_settings
        assert d["initial_latent_sha256"] == initial_hashes
        latents[arm] = torch.load(d["terminal_latents"], map_location="cpu",
                                  weights_only=True)
        assert set(latents[arm]) == {"video_latents", "audio_latents"}
        assert all(torch.isfinite(x).all().item() for x in latents[arm].values())
        denoise = float(d["denoise_wall_seconds"])
        decode = float(v["decode_wall_seconds"])
        dit_sum = sum(x["dit_seconds"] for x in d["steps"])
        assert math.isclose(dit_sum, d["sum_dit_seconds"], rel_tol=1e-9)
        rows[arm] = {"denoise_wall_seconds": denoise,
                     "decode_wall_seconds": decode,
                     "total_wall_seconds": denoise + decode,
                     "sum_dit_seconds": dit_sum,
                     "mean_dit_seconds_steps_1_to_19": sum(
                         x["dit_seconds"] for x in d["steps"][1:]) / 19,
                     "median_dit_seconds_steps_1_to_19": statistics.median(
                         x["dit_seconds"] for x in d["steps"][1:]),
                     "first_dit_seconds": d["steps"][0]["dit_seconds"],
                     "peak_allocated_gib": d["peak_allocated_gib"],
                     "main_tokens": d["steps"][0]["main_tokens"],
                     "video": v["video"]}
    for arm in MEASURE_ARMS:
        rows[arm]["total_speedup_vs_warm_bf16"] = (
            rows["bf16_warm"]["total_wall_seconds"] / rows[arm]["total_wall_seconds"])
        rows[arm]["dit_speedup_vs_warm_bf16"] = (
            rows["bf16_warm"]["sum_dit_seconds"] / rows[arm]["sum_dit_seconds"])
        rows[arm]["total_slowdown_vs_sage3"] = (
            rows[arm]["total_wall_seconds"] / rows["sage3_fixed"]["total_wall_seconds"])
        rows[arm]["dit_slowdown_vs_sage3"] = (
            rows[arm]["sum_dit_seconds"] / rows["sage3_fixed"]["sum_dit_seconds"])
        rows[arm]["terminal_latent_nmse_vs_bf16"] = {
            name: float((latents[arm][name].float() - latents["bf16"][name].float())
                        .square().sum() / latents["bf16"][name].float().square().sum())
            for name in ("video_latents", "audio_latents")}
    assert all(torch.equal(latents["bf16"][name], latents["bf16_warm"][name])
               for name in ("video_latents", "audio_latents"))
    rows["split_mixed"]["terminal_latent_nmse_vs_sage3"] = {
        name: float((latents["split_mixed"][name].float() -
                     latents["sage3_fixed"][name].float()).square().sum() /
                    latents["sage3_fixed"][name].float().square().sum())
        for name in ("video_latents", "audio_latents")}
    report = {"format": "e121-h3-end-to-end-summary-v1", "case": args.case,
              "prompt_id": preflight["prompt_id"], "seed": preflight["seed"],
              "settings": common_settings, "initial_latent_sha256": initial_hashes,
              "preflight_nmse": {arm: preflight["arms"][arm].get("vs_bf16_dit_nmse")
                                 for arm in PREFLIGHT_ARMS[1:]},
              "arms": rows, "timing_note": "Each arm sequentially ran full denoise and separate VAE decode on one GPU; total includes both phase model loads and media writing. The first BF16 run is a cold-cache reference; bf16_warm is the fair speed baseline."}
    path = folder / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
