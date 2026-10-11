#!/usr/bin/env python3
"""Resumable multi-prompt full-timestep raw DiT cache collector for H3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt, serialize_call


class CapturedAllSteps(RuntimeError):
    """Intentional early exit: raw DiT cache does not need VAE decoding."""


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def complete(directory: Path, prompt_id: int, steps: int) -> bool:
    path = directory / f"p{prompt_id}" / "manifest.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        return data.get("prompt_id") == prompt_id and len(data.get("samples", [])) == steps and all(
            (path.parent / row["file"]).is_file() for row in data["samples"])
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def main() -> None:
    a = args(); cfg = json.loads(a.config.read_text())
    ids, steps, captured = cfg["prompt_ids"], int(cfg["steps"]), list(cfg["capture_steps"])
    if captured != list(range(steps)):
        raise ValueError("this collector requires every scheduler call to be captured")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "contract.json").write_text(json.dumps({**cfg, "format": "minimax-h3-standard-raw-dit-cache-v1"}, indent=2))
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    for prompt_id in ids:
        target_dir = a.output_dir / f"p{prompt_id}"
        if complete(a.output_dir, prompt_id, steps):
            print(f"p{prompt_id}: complete; reuse", flush=True); continue
        target_dir.mkdir(parents=True, exist_ok=True)
        row, saved, call_id = read_prompt(prompt_id), [], 0
        def capture(_module, call_args, call_kwargs):
            nonlocal call_id
            target = target_dir / f"sample_p{prompt_id}_s{call_id:02d}.pt"
            payload = serialize_call(call_args, call_kwargs)
            payload["meta"] = {"prompt_id": prompt_id, "seed": int(row["seed"]), "step": call_id}
            torch.save(payload, target)
            saved.append({**payload["meta"], "file": target.name})
            call_id += 1
            # The pipeline otherwise proceeds into video/audio VAE decoding
            # after the last denoising call.  It adds minutes per prompt yet
            # contributes nothing to a raw DiT-call calibration cache.
            if call_id == steps:
                raise CapturedAllSteps()
        handle = pipe.dit.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            pipe(prompt=row["prompt"], seed=int(row["seed"]), height=cfg["shape"]["height"], width=cfg["shape"]["width"],
                 num_frames=cfg["shape"]["frames"], num_inference_steps=steps, cfg_scale=1.0, tiled=True)
        except CapturedAllSteps:
            pass
        finally:
            handle.remove()
        if call_id != steps:
            raise RuntimeError(f"p{prompt_id}: captured {call_id}/{steps}")
        (target_dir / "manifest.json").write_text(json.dumps({
            "format": "minimax-h3-standard-raw-dit-cache-v1", "prompt_id": prompt_id,
            "capture_steps": captured, "height": cfg["shape"]["height"], "width": cfg["shape"]["width"],
            "frames": cfg["shape"]["frames"], "inference_steps": steps, "samples": saved,
        }, indent=2))
        print(f"p{prompt_id}: complete ({len(saved)} calls)", flush=True)


if __name__ == "__main__":
    main()
