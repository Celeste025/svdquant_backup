#!/usr/bin/env python3
"""Generate a BF16, plain W4A4, or standard SVDQuant MiniMax-H3 video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffsynth.utils.data.audio_video import write_video_audio
from minimax_h3_svdquant_common import install_runtime_hooks, load_h3_pipeline, nvfp4_qdq, read_prompt, target_linears


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["bf16", "w4a4", "svdquant"], required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-id", type=int, required=True)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def _install(linear, qweight, smooth=None, a=None, b=None):
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
def apply_plain(dit):
    hooks = []
    for i, (_, linear) in enumerate(target_linears(dit), 1):
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        hooks.append(_install(linear, nvfp4_qdq(weight)))
        del weight
        if i % 20 == 0: print(f"plain W4A4: {i}/200", flush=True)
        torch.cuda.empty_cache()
    return hooks


@torch.no_grad()
def apply_svdquant(dit, path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("invalid standard H3 SVDQuant state")
    hooks = []
    for i, (name, linear) in enumerate(target_linears(dit), 1):
        layer = state["layers"][name]
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        smooth, a, b = layer["smooth"].to(weight), layer["final_a"].to(weight), layer["final_b"].to(weight)
        hooks.append(_install(linear, nvfp4_qdq(weight * smooth - b @ a), layer["smooth"], layer["final_a"], layer["final_b"]))
        del weight, smooth, a, b
        if i % 20 == 0: print(f"SVDQuant reload: {i}/200", flush=True)
        torch.cuda.empty_cache()
    return hooks


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("invalid H3 output shape")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = read_prompt(args.prompt_id)
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    hooks = []
    if args.case != "bf16":
        pipe.load_models_to_device(["dit"])
        hooks = apply_plain(pipe.dit) if args.case == "w4a4" else apply_svdquant(pipe.dit, args.state)
    video, audio = pipe(prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
                        num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
                        rand_device="cpu", tiled=True)
    if len(video) != args.frames or (hooks and any(h.act.calls == 0 for h in hooks)):
        raise RuntimeError("generation or activation-hook check failed")
    target = args.output_dir / f"{args.case}_id{args.prompt_id}_seed{row['seed']}_{args.height}x{args.width}_{args.frames}f_{args.steps}steps.mp4"
    write_video_audio(video=video, audio=audio, output_path=str(target), fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    (args.output_dir / f"{args.case}.json").write_text(json.dumps({"case": args.case, "prompt_id": args.prompt_id,
        "seed": int(row["seed"]), "height": args.height, "width": args.width, "frames": len(video), "steps": args.steps,
        "output": str(target)}, indent=2, ensure_ascii=False))
    print(f"saved {target}", flush=True)


if __name__ == "__main__":
    main()
