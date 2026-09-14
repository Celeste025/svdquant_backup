#!/usr/bin/env python3
"""Collect exactly 2 prompts x 4 H3 DiT calls for the PTQ smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt, serialize_call


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/home/wjq/workspace/svdquant-exp/results/calib/minimax_h3_smoke_2p8s"))
    p.add_argument("--prompt-ids", type=int, nargs=2, default=[48, 105])
    p.add_argument("--capture-steps", type=int, nargs=4, default=[0, 10, 20, 29])
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--frames", type=int, default=39)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps != 30 or sorted(args.capture_steps) != [0, 10, 20, 29]:
        raise ValueError("Smoke calibration requires 30 steps and captures 0/10/20/29")
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("height/width must be multiples of 32; frames must satisfy 17n+5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("sample_*.pt"))
    if existing and not args.overwrite:
        if len(existing) == 8:
            print(f"Calibration already complete: {args.output_dir}", flush=True)
            return
        raise RuntimeError(f"Partial cache exists ({len(existing)} files); use --overwrite")
    for path in existing:
        path.unlink()

    pipe = load_h3_pipeline(full=True)
    manifest = {
        "format": "minimax-h3-raw-dit-cache-v1",
        "prompt_ids": args.prompt_ids,
        "capture_steps": args.capture_steps,
        "height": args.height, "width": args.width, "frames": args.frames, "inference_steps": args.steps,
        "samples": [],
    }
    for prompt_id in args.prompt_ids:
        row = read_prompt(prompt_id)
        call_id = 0

        def capture(_module, call_args, call_kwargs):
            nonlocal call_id
            if call_id in args.capture_steps:
                index = len(manifest["samples"])
                target = args.output_dir / f"sample_{index:02d}_p{prompt_id}_s{call_id:02d}.pt"
                payload = serialize_call(call_args, call_kwargs)
                payload["meta"] = {"prompt_id": prompt_id, "seed": int(row["seed"]), "step": call_id}
                torch.save(payload, target)
                manifest["samples"].append({**payload["meta"], "file": target.name})
                print(f"saved {target.name}", flush=True)
            call_id += 1

        handle = pipe.dit.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            pipe(
                prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
                num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0, tiled=True,
            )
        finally:
            handle.remove()
        if call_id != args.steps:
            raise RuntimeError(f"prompt {prompt_id}: expected {args.steps} DiT calls, observed {call_id}")

    if len(manifest["samples"]) != 8:
        raise RuntimeError(f"Expected exactly 8 samples, captured {len(manifest['samples'])}")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Calibration complete: {args.output_dir} (8 raw DiT calls)", flush=True)


if __name__ == "__main__":
    main()
