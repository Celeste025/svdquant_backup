#!/usr/bin/env python3
"""Controlled diagnostic: scale rCM-Wan velocity error toward its BF16 trace.

This is deliberately an oracle diagnostic, not a deployable method.  At each
step it takes the W4A4 velocity evaluated on its own rollout state and mixes
it with the paired BF16-reference velocity:

    v_alpha = v_bf16_trace + alpha * (v_w4a4 - v_bf16_trace)

Thus alpha**2 is the exact retained NMSE relative to the *reported* paired
rollout-velocity target at that step.  It cleanly tests whether a prescribed
per-step v error reduction transfers to final latent/video similarity.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import WanPipeline

from eval_rcm_restore_video_similarity import compare as compare_video
from exp_rcm_qk_bf16_oracle import build_bf16_reference, decode_and_save, tensor_metrics
from exp_rcm_restore_top_fraction_bf16 import CKPT, MODEL, PROMPT, load_helper

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run_case(alpha: float, reference: dict[str, Any], out: Path, helper: Any) -> tuple[Path, list[dict[str, Any]]]:
    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    helper.load_quantized_transformer(pipe, CKPT, MODEL)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    embeds = reference["prompt_embeds"].to(device=device, dtype=pipe.transformer.dtype)
    times = torch.tensor(reference["times"], device=device, dtype=torch.float64)
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    latents = reference["initial_latent"].to(device=device, dtype=torch.float64).clone()
    rows = []
    for step, (current, nxt) in enumerate(zip(times[:-1], times[1:], strict=True)):
        timestep = (current.float() * ones * 1000).to(dtype=pipe.transformer.dtype)
        raw_v = pipe.transformer(hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                                 encoder_hidden_states=embeds, return_dict=False)[0].to(torch.float64)
        bf_trace_v = reference["velocities"][step].to(device=device, dtype=torch.float64)
        mixed_v = bf_trace_v + alpha * (raw_v - bf_trace_v)
        raw = tensor_metrics(raw_v, bf_trace_v)
        mixed = tensor_metrics(mixed_v, bf_trace_v)
        latents = (1 - nxt) * (latents - current * mixed_v) + nxt * reference["noises"][step].to(device=device, dtype=torch.float64)
        latent = tensor_metrics(latents, reference["latents"][step + 1])
        rows.append({"alpha": alpha, "nominal_v_nmse_retention": alpha * alpha, "step": step,
                     "timestep": float(current),
                     **{f"raw_{k}": v for k, v in raw.items()},
                     **{f"mixed_{k}": v for k, v in mixed.items()},
                     **{f"latent_{k}": v for k, v in latent.items()}})
    video = out / f"alpha{alpha:.3f}.mp4"
    decode_and_save(pipe, helper, latents, video)
    pipe.to("cpu")
    del pipe, embeds, latents
    gc.collect(); torch.cuda.empty_cache()
    return video, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.8660254, 0.5, 0.0])
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/samples/rcm_velocity_error_oracle/airplane_seed303")
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    helper = load_helper()
    bf16 = out / "bf16.mp4"
    reference, _, _ = build_bf16_reference(args.prompt, args.seed, args.height, args.width, args.frames, bf16, helper)
    rows, videos = [], {}
    for alpha in args.alphas:
        print(f"[run] alpha={alpha:.7g}, nominal v-NMSE reduction={1-alpha*alpha:.1%}", flush=True)
        video, case_rows = run_case(alpha, reference, out, helper)
        videos[alpha] = video; rows.extend(case_rows)
    write_csv(out / "per_step_metrics.csv", rows)
    video_rows = []
    for alpha, video in videos.items():
        metrics = compare_video(bf16, video, torch.device("cuda"), include_temporal=False)
        video_rows.append({"alpha": alpha, "nominal_v_nmse_reduction": 1 - alpha * alpha,
                           "reference": str(bf16), "video": str(video), **metrics})
        torch.cuda.empty_cache()
    write_csv(out / "video_metrics.csv", video_rows)
    write_json(out / "config.json", {"prompt": args.prompt, "seed": args.seed, "alphas": args.alphas,
        "definition": "Oracle interpolation to paired BF16 trace velocity; alpha=1 is NVFP4, alpha=0 exactly reconstructs the BF16 trajectory.",
        "schedule": "rCM TrigFlow->RectifiedFlow, 4 steps, guidance=0"})
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
