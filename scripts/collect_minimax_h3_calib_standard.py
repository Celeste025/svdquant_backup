#!/usr/bin/env python3
"""Capture standard MiniMax-H3 raw DiT calibration calls at real evaluation shape."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt, serialize_call


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-id", type=int, required=True)
    p.add_argument("--capture-steps", type=int, nargs="+", required=True)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("H3 requires H/W multiples of 32 and frames=17n+5")
    capture_steps = sorted(set(args.capture_steps))
    if not capture_steps or capture_steps[0] < 0 or capture_steps[-1] >= args.steps:
        raise ValueError("capture steps must be unique indices in [0, steps)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = read_prompt(args.prompt_id)
    saved: list[dict] = []
    call_id = 0

    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)

    def capture(_module, call_args, call_kwargs):
        nonlocal call_id
        if call_id in capture_steps:
            target = args.output_dir / f"sample_p{args.prompt_id}_s{call_id:02d}.pt"
            payload = serialize_call(call_args, call_kwargs)
            payload["meta"] = {"prompt_id": args.prompt_id, "seed": int(row["seed"]), "step": call_id}
            torch.save(payload, target)
            saved.append({**payload["meta"], "file": target.name})
            print(f"saved {target.name}", flush=True)
        call_id += 1

    handle = pipe.dit.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        pipe(prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
             num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0, tiled=True)
    finally:
        handle.remove()
    if call_id != args.steps or len(saved) != len(capture_steps):
        raise RuntimeError(f"captured {len(saved)}/{len(capture_steps)} at {call_id}/{args.steps} DiT calls")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "format": "minimax-h3-standard-raw-dit-cache-v1", "prompt_id": args.prompt_id,
        "capture_steps": capture_steps, "height": args.height, "width": args.width,
        "frames": args.frames, "inference_steps": args.steps, "samples": saved,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
