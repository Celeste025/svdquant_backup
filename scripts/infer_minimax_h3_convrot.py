#!/usr/bin/env python3
"""Reload a real ConvRot NVFP4 H3 state and generate a matched video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffsynth.utils.data.audio_video import write_video_audio
from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt
from minimax_h3_convrot_common import assert_target_structure, configure_offloaded_quant_linear, import_convrot, state_config, unpack_weight, warmup_convrot_extension


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-id", type=int, required=True)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def apply_state(dit, path: Path):
    import_convrot()
    warmup_convrot_extension()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-convrot-nvfp4-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("invalid ConvRot H3 state")
    from convrot.rotated_nvfp4_tensor import RotatedNVFP4Tensor
    targets = assert_target_structure(dit)
    for index, (name, linear) in enumerate(targets, start=1):
        weight = unpack_weight(state["layers"][name], "cuda")
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if not isinstance(linear.weight, RotatedNVFP4Tensor):
            raise RuntimeError(f"{name}: state did not restore RotatedNVFP4Tensor")
        configure_offloaded_quant_linear(linear)
        if index % 20 == 0:
            print(f"ConvRot restored {index}/200", flush=True)
    return state


def main():
    args = parse_args()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("invalid H3 output shape")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = read_prompt(args.prompt_id)
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    state = apply_state(pipe.dit, args.state)
    video, audio = pipe(prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
                        num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
                        rand_device="cpu", tiled=True)
    if len(video) != args.frames:
        raise RuntimeError("unexpected generated frame count")
    target = args.output_dir / f"convrot_id{args.prompt_id}_seed{row['seed']}_{args.height}x{args.width}_{args.frames}f_{args.steps}steps.mp4"
    write_video_audio(video=video, audio=audio, output_path=str(target), fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    meta = {"case": "convrot", "prompt_id": int(args.prompt_id), "seed": int(row["seed"]),
            "height": args.height, "width": args.width, "frames": len(video), "steps": args.steps,
            "state": str(args.state), "config": state_config(), "output": str(target)}
    (args.output_dir / "convrot.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"saved {target}", flush=True)


if __name__ == "__main__":
    main()
