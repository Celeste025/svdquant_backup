#!/usr/bin/env python3
"""Generate H3 SVDQuant videos with its highest independent-error layers restored."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffsynth.utils.data.audio_video import write_video_audio
from minimax_h3_svdquant_common import install_runtime_hooks, load_h3_pipeline, nvfp4_qdq, read_prompt, target_linears


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-id", type=int, required=True)
    p.add_argument("--restore-pct", type=int, choices=(10, 20), required=True)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def install_svdq(linear, qweight, smooth, a, b):
    linear.load_state_dict({"weight": qweight.detach().cpu()}, assign=True, strict=False)
    linear.disk_offload = False
    linear.offload_dtype = linear.onload_dtype = torch.bfloat16
    linear.offload_device = linear.onload_device = torch.device("cpu")
    linear.computation_dtype = torch.bfloat16
    linear.computation_device = torch.device("cuda")
    linear.vram_limit = 0.0
    linear.state = 1
    runtime = install_runtime_hooks(linear, smooth, a, b)
    runtime.branch.to(device="cuda", dtype=torch.bfloat16)
    return runtime


@torch.no_grad()
def restore_bf16(linear):
    weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
    linear.load_state_dict({"weight": weight.detach().cpu()}, assign=True, strict=False)
    linear.disk_offload = False
    linear.offload_dtype = linear.onload_dtype = torch.bfloat16
    linear.offload_device = linear.onload_device = torch.device("cpu")
    linear.computation_dtype = torch.bfloat16
    linear.computation_device = torch.device("cuda")
    linear.vram_limit = 0.0
    linear.state = 1


@torch.no_grad()
def apply(dit, state_path: Path, restore_pct: int):
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    layers = state.get("layers", {})
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(layers) != 200:
        raise RuntimeError("invalid H3 SVDQuant state")
    count = 200 * restore_pct // 100
    restore = {name for name, _ in sorted(layers.items(), key=lambda item: item[1]["final_error"], reverse=True)[:count]}
    runtimes = {}
    for index, (name, linear) in enumerate(target_linears(dit), 1):
        item = layers[name]
        if name in restore:
            restore_bf16(linear)
        else:
            weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
            smooth, a, b = item["smooth"].to(weight), item["final_a"].to(weight), item["final_b"].to(weight)
            runtimes[name] = install_svdq(linear, nvfp4_qdq(weight * smooth - b @ a), item["smooth"], item["final_a"], item["final_b"])
        if index % 20 == 0:
            print(f"prepared {index}/200", flush=True)
    return restore, runtimes


def main():
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = read_prompt(args.prompt_id)
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    restore, runtimes = apply(pipe.dit, args.state, args.restore_pct)
    video, audio = pipe(prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
                        num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
                        rand_device="cpu", tiled=True)
    if len(video) != args.frames or any(runtime.act.calls == 0 for runtime in runtimes.values()):
        raise RuntimeError("generation or remaining SVDQuant-hook validation failed")
    target = args.output_dir / f"restore_top{args.restore_pct}pct_id{args.prompt_id}_seed{row['seed']}_{args.height}x{args.width}_{args.frames}f_{args.steps}steps.mp4"
    write_video_audio(video=video, audio=audio, output_path=str(target), fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    (args.output_dir / f"restore_top{args.restore_pct}pct.json").write_text(json.dumps({
        "prompt_id": args.prompt_id, "seed": int(row["seed"]), "restore_pct": args.restore_pct,
        "restore_count": len(restore), "restored_layers": sorted(restore), "output": str(target),
    }, indent=2))
    print(f"saved {target}", flush=True)


if __name__ == "__main__":
    main()
