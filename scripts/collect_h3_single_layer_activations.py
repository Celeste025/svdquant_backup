#!/usr/bin/env python3
"""Capture one H3 linear layer's input activation at every denoising step.

Runs the bf16 pruned pipeline on a few VBench prompts and stores the full
flattened input of a single target linear layer per step.  Payloads feed
scripts/analyze_h3_deltaquant_vs_svdquant.py, which compares SVDQuant-style
and DeltaQuant-style activation quantization offline on the same activations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import DATA_ROOT, load_h3_pipeline

CASES_DEFAULT = ["vbench_001", "vbench_076", "vbench_166"]
LAYER_DEFAULT = "blocks.24.mlp.fc2"
METADATA = DATA_ROOT / "videos/svdquant-videoeval-minimax-h3/metadata/videos.jsonl"
OUTPUT_DIR_DEFAULT = DATA_ROOT / "runs/h3-singlelayer-dq-vs-svdq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=CASES_DEFAULT)
    parser.add_argument("--layer", default=LAYER_DEFAULT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=39)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--reserve-gib", type=float, default=4.0,
                        help="VRAM headroom kept free; raise it to reduce disk-offload thrash on a shared card")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_prompts(cases: list[str]) -> dict[str, dict]:
    prompts: dict[str, dict] = {}
    for line in METADATA.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        sidecar = row.get("sidecar") or {}
        case_id = sidecar.get("case_id")
        if case_id in cases and case_id not in prompts and sidecar.get("prompt"):
            prompts[case_id] = {"prompt": sidecar["prompt"], "seed": int(sidecar.get("seed", 0))}
    missing = [case for case in cases if case not in prompts]
    if missing:
        raise KeyError(f"missing prompts for {missing}")
    return prompts


def compute_layout(pipe, prompt: str, height: int, width: int, frames: int) -> dict[str, list[int]]:
    """Boundaries of the packed sequence: [text | audio | video | pad]."""
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va

    text_len = len(presentation_t2va(pipe.tokenizer, prompt)[0])
    latent_t = ((frames - 5) // 17) * 5 + 2
    frame_rows = (height // 16 // 2) * (width // 16 // 2)
    audio_rows = round(frames / 24.0 * 40.0) * 2
    video_rows = latent_t * frame_rows
    used = text_len + audio_rows + video_rows
    seq_len = ((used + 63) // 64) * 64
    return {
        "text": [0, text_len],
        "audio": [text_len, text_len + audio_rows],
        "video": [text_len + audio_rows, used],
        "pad": [used, seq_len],
    }


def resolve_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module = root
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        case for case in args.cases
        if args.overwrite or not (args.output_dir / f"{case}.pt").is_file()
    ]
    if not pending:
        print(f"all cases already captured in {args.output_dir}; nothing to do", flush=True)
        return

    pipe = load_h3_pipeline(full=True, reserve_gib=args.reserve_gib)
    print("pipeline loaded", flush=True)
    module = resolve_module(pipe.dit, args.layer)
    print(f"target layer: {args.layer}", flush=True)

    for case in pending:
        row = prompts[case]
        state = {"step": -1, "timesteps": []}
        captures: dict[int, torch.Tensor] = {}

        def capture_input(_module, inputs):
            step = state["step"]
            x = inputs[0]
            if x.dim() != 2:
                x = x.reshape(-1, x.shape[-1])
            captures[step] = x.detach().to("cpu")

        handle = module.register_forward_pre_hook(capture_input)

        def step_counter(_module, _inputs, kwargs):
            state["step"] += 1
            t = kwargs.get("unique_timesteps")
            state["timesteps"].append(None if t is None else t.detach().cpu())

        counter = pipe.dit.register_forward_pre_hook(step_counter, with_kwargs=True)
        try:
            pipe(
                prompt=row["prompt"], seed=row["seed"], height=args.height, width=args.width,
                num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
                rand_device="cpu", tiled=True,
            )
        finally:
            handle.remove()
            counter.remove()

        captured = sorted(captures)
        if captured != list(range(args.steps)):
            raise RuntimeError(f"{case}: expected steps 0..{args.steps - 1}, captured {captured}")
        layout = compute_layout(pipe, row["prompt"], args.height, args.width, args.frames)
        grid = {
            "t": ((args.frames - 5) // 17) * 5 + 2,
            "h": args.height // 16 // 2,
            "w": args.width // 16 // 2,
        }
        payload = {
            "case": case,
            "layer": args.layer,
            "prompt": row["prompt"],
            "seed": row["seed"],
            "config": {"height": args.height, "width": args.width, "frames": args.frames, "steps": args.steps},
            "layout": layout,
            "grid": grid,
            "timesteps": state["timesteps"],
            "steps": captures,
        }
        target = args.output_dir / f"{case}.pt"
        torch.save(payload, target)
        x0 = captures[0]
        print(f"[{case}] saved {target.name}; steps={len(captured)}; x0={tuple(x0.shape)} {x0.dtype}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
