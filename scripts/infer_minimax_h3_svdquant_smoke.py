#!/usr/bin/env python3
"""Reload BF16 H3, apply BF16/plain-W4A4/SVDQuant state, and generate ID 42."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from diffsynth.utils.data.audio_video import write_video_audio
from minimax_h3_svdquant_common import (
    install_runtime_hooks, load_h3_pipeline, nvfp4_qdq, read_prompt,
    target_linears, validate_state_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["bf16", "w4a4", "svdquant"], required=True)
    p.add_argument("--state", type=Path, default=Path("/home/wjq/workspace/svdquant-exp/results/checkpoints/minimax_h3_svdquant_smoke/quant_state.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("/home/wjq/workspace/svdquant-exp/results/samples/minimax_h3_svdquant_smoke/id42_seed52386"))
    p.add_argument("--prompt-id", type=int, default=42)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=39)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def _install_cpu_qweight(linear, qweight, smooth=None, a=None, b=None):
    linear.load_state_dict({"weight": qweight.detach().cpu()}, assign=True, strict=False)
    linear.disk_offload = False
    linear.offload_dtype = linear.onload_dtype = torch.bfloat16
    linear.offload_device = linear.onload_device = torch.device("cpu")
    linear.computation_dtype = torch.bfloat16
    linear.computation_device = torch.device("cuda")
    linear.vram_limit = 0.0
    linear.state = 1
    runtime = install_runtime_hooks(linear, smooth, a, b)
    if runtime.branch is not None:
        runtime.branch.to(device="cuda", dtype=torch.bfloat16)
    return runtime


@torch.no_grad()
def apply_plain_w4a4(dit) -> list:
    hooks = []
    for index, (name, linear) in enumerate(target_linears(dit)):
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        qweight = nvfp4_qdq(weight)
        hooks.append(_install_cpu_qweight(linear, qweight))
        del weight, qweight
        torch.cuda.empty_cache()
        if (index + 1) % 20 == 0:
            print(f"plain W4A4: {index + 1}/200", flush=True)
    return hooks


@torch.no_grad()
def apply_svdquant(dit, state_path: Path) -> list:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_state_manifest(state, 200)
    hooks = []
    for index, (name, linear) in enumerate(target_linears(dit)):
        layer = state["layers"][name]
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        smooth = layer["smooth"].to(weight)
        a = layer["a"].to(weight)
        b = layer["b"].to(weight)
        qweight = nvfp4_qdq(weight * smooth - b @ a)
        hooks.append(_install_cpu_qweight(linear, qweight, layer["smooth"], layer["a"], layer["b"]))
        del weight, smooth, a, b, qweight
        torch.cuda.empty_cache()
        if (index + 1) % 20 == 0:
            print(f"SVDQuant reload: {index + 1}/200", flush=True)
    return hooks


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("Invalid H3 smoke shape")
    row = read_prompt(args.prompt_id)
    seed = int(row["seed"]) if args.seed is None else args.seed
    if args.prompt_id == 42 and seed != 52386:
        raise ValueError("ID 42 smoke must use its original seed 52386")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    runtime_hooks = []
    if args.case != "bf16":
        pipe.load_models_to_device(["dit"])
        if args.case == "w4a4":
            runtime_hooks = apply_plain_w4a4(pipe.dit)
        else:
            runtime_hooks = apply_svdquant(pipe.dit, args.state)
        if len(runtime_hooks) != 200:
            raise RuntimeError("Expected 200 activation QDQ hooks")

    # All processes use the same CPU RNG path, original seed and scheduler.
    video, audio = pipe(
        prompt=row["prompt"], seed=seed, height=args.height, width=args.width,
        num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
        rand_device="cpu", tiled=True,
    )
    if len(video) != args.frames:
        raise RuntimeError(f"Expected {args.frames} frames, got {len(video)}")
    if args.case != "bf16" and any(h.act.calls == 0 for h in runtime_hooks):
        missing = sum(h.act.calls == 0 for h in runtime_hooks)
        raise RuntimeError(f"{missing} activation QDQ hooks never fired")
    output = args.output_dir / f"{args.case}_id{args.prompt_id}_seed{seed}_{args.height}x{args.width}_{args.frames}f_{args.steps}steps.mp4"
    write_video_audio(video=video, audio=audio, output_path=str(output), fps=24,
                      audio_sample_rate=pipe.audio_vae.sample_rate)
    meta = {
        "case": args.case, "prompt_id": args.prompt_id, "prompt": row["prompt"], "seed": seed,
        "height": args.height, "width": args.width, "frames": len(video), "steps": args.steps,
        "shared_noise_contract": "same CPU RNG implementation and seed for video/audio initial noise",
        "output": str(output), "audio_samples": int(audio.shape[-1]),
    }
    (args.output_dir / f"{args.case}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
