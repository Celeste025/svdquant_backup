#!/usr/bin/env python3
"""Measure single-call BF16 vs SVDQuant error along one MiniMax-H3 trajectory.

The quantized model is *teacher forced*: each invocation receives precisely the
arguments seen by BF16 at that denoising step.  It therefore measures local
DiT error rather than the (usually much larger) error after a divergent
sampling trajectory has fed back into later steps.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch

from minimax_h3_svdquant_common import (install_runtime_hooks, load_h3_pipeline, nvfp4_qdq,
                                        read_prompt, target_linears, tree_cpu, tree_device)


@torch.no_grad()
def apply_svdquant(dit: torch.nn.Module, path: Path):
    """Install precisely the standard-inference SVDQuant state, without video I/O deps."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("invalid standard H3 SVDQuant state")
    hooks = []
    for i, (name, linear) in enumerate(target_linears(dit), 1):
        layer = state["layers"][name]
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        smooth = layer["smooth"].to(weight)
        a, b = layer["final_a"].to(weight), layer["final_b"].to(weight)
        linear.load_state_dict({"weight": nvfp4_qdq(weight * smooth - b @ a).detach().cpu()}, assign=True, strict=False)
        linear.disk_offload = False
        linear.offload_dtype = linear.onload_dtype = torch.bfloat16
        linear.offload_device = linear.onload_device = torch.device("cpu")
        linear.computation_dtype, linear.computation_device = torch.bfloat16, torch.device("cuda")
        linear.vram_limit, linear.state = 0.0, 1
        runtime = install_runtime_hooks(linear, smooth, a, b)
        if runtime.branch is not None:
            runtime.branch.to(device="cuda", dtype=torch.bfloat16)
        hooks.append(runtime)
        del weight, smooth, a, b
        if i % 20 == 0:
            print(f"SVDQuant reload: {i}/200", flush=True)
        torch.cuda.empty_cache()
    return hooks


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--prompt-id", type=int, default=16)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def split_output(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple) or len(output) != 2 or not all(torch.is_tensor(x) for x in output):
        raise TypeError(f"unexpected DiT output: {type(output)!r}")
    return output


def metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    p, r = pred.float(), ref.float()
    diff = p - r
    mse = diff.square().mean(dtype=torch.float64)
    ref_mse = r.square().mean(dtype=torch.float64).clamp_min(1e-30)
    cosine = (p.flatten().double() @ r.flatten().double()) / (
        p.flatten().double().norm() * r.flatten().double().norm()
    ).clamp_min(1e-30)
    return {"max_abs": float(diff.abs().max().item()), "mse": float(mse.item()),
            "nmse": float((mse / ref_mse).item()), "rmse": float(mse.sqrt().item()),
            "cosine": float(cosine.item())}


@torch.inference_mode()
def capture_bf16(args: argparse.Namespace, row: dict[str, Any]) -> list[dict[str, Any]]:
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    calls: list[dict[str, Any]] = []

    def hook(_module, input_args, input_kwargs, output):
        video, audio = split_output(output)
        calls.append({"args": tree_cpu(input_args), "kwargs": tree_cpu(input_kwargs),
                      "video": video.detach().cpu(), "audio": audio.detach().cpu()})

    handle = pipe.dit.register_forward_hook(hook, with_kwargs=True)
    try:
        pipe(prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
             num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
             rand_device="cpu", tiled=True)
    finally:
        handle.remove()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if len(calls) != args.steps:
        raise RuntimeError(f"expected {args.steps} DiT calls, captured {len(calls)}")
    return calls


@torch.inference_mode()
def main() -> None:
    args = args_parser()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("invalid H3 output shape")
    row = read_prompt(args.prompt_id)
    print("capturing BF16 sampling trajectory", flush=True)
    calls = capture_bf16(args, row)
    print("loading SVDQuant DiT", flush=True)
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    hooks = apply_svdquant(pipe.dit, args.state)
    results = []
    for step, call in enumerate(calls):
        output = pipe.dit(*tree_device(call["args"], "cuda"), **tree_device(call["kwargs"], "cuda"))
        video, audio = split_output(output)
        item = {"step": step, "timestep": float(call["kwargs"]["unique_timesteps"].flatten()[0]),
                "video": metrics(video, call["video"].to("cuda")),
                "audio": metrics(audio, call["audio"].to("cuda"))}
        results.append(item)
        print(f"step={step:02d} t={item['timestep']:.6f} video_nmse={item['video']['nmse']:.6g} "
              f"video_cos={item['video']['cosine']:.8f}", flush=True)
    if any(h.act.calls == 0 for h in hooks):
        raise RuntimeError("one or more quantization hooks were not exercised")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"definition": "teacher-forced per-DiT-call error: quantized and BF16 receive identical BF16 trajectory inputs",
               "prompt_id": args.prompt_id, "seed": int(row["seed"]), "prompt": row["prompt"],
               "height": args.height, "width": args.width, "frames": args.frames, "steps": args.steps,
               "results": results}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "timestep", "video_nmse", "video_mse", "video_rmse", "video_max_abs", "video_cosine", "audio_nmse", "audio_mse", "audio_rmse", "audio_max_abs", "audio_cosine"])
        writer.writeheader()
        for x in results:
            writer.writerow({"step": x["step"], "timestep": x["timestep"],
                             **{f"video_{k}": v for k, v in x["video"].items()},
                             **{f"audio_{k}": v for k, v in x["audio"].items()}})
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
