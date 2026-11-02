#!/usr/bin/env python3
"""E103: save complete paired H3 DiT outputs for error-pattern analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_minimax_h3_paired_step_error import apply_svdquant, metrics, split_output
from minimax_h3_svdquant_common import load_h3_pipeline, tree_device


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--calls", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--e101-report", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected-prompt-id", type=int, default=16)
    p.add_argument("--expected-seed", type=int, default=20026)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    calls = torch.load(args.calls, map_location="cpu", weights_only=False)
    expected = json.loads(args.e101_report.read_text())
    if len(calls) != 20 or len(expected["results"]) != 20 or expected["variant"] != "svdquant" \
            or (expected["prompt_id"], expected["seed"]) != (args.expected_prompt_id, args.expected_seed):
        raise RuntimeError("E101 case identity or step count mismatch")
    grid = (37, 18, 32)
    if any(tuple(c["video"].shape) != (math.prod(grid), 96) or
           tuple(c["audio"].shape) != (414, 32) or
           c["kwargs"]["img_pos_for_infer_output_info"]["position_ids"].numel() != math.prod(grid)
           for c in calls):
        raise RuntimeError("unexpected output shape; video grid mapping invalid")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    hooks = apply_svdquant(pipe.dit, args.state)
    records = []
    for step, call in enumerate(calls):
        target = args.raw_dir / f"step_{step:02d}.pt"
        if target.exists():
            payload = torch.load(target, map_location="cpu", weights_only=False)
            if payload.get("format") != "h3-e103-paired-output-v1" or payload.get("step") != step:
                raise RuntimeError(f"invalid existing step file: {target}")
            print(f"reused {target}", flush=True)
        else:
            video, audio = split_output(pipe.dit(*tree_device(call["args"], "cuda"),
                                                   **tree_device(call["kwargs"], "cuda")))
            ref_video = call["video"].to("cuda")
            ref_audio = call["audio"].to("cuda")
            vm = metrics(video, ref_video)
            am = metrics(audio, ref_audio)
            old = expected["results"][step]
            if old["step"] != step or any(abs(m["nmse"] - old[k]["nmse"]) > 1e-6
                                          for k, m in (("video", vm), ("audio", am))):
                raise RuntimeError(f"step {step}: paired NMSE differs from E101 baseline")
            payload = {
                "format": "h3-e103-paired-output-v1", "step": step,
                "unique_timesteps": call["kwargs"]["unique_timesteps"].clone(),
                "video_grid_thw": grid, "video_row_order": "t,h,w (patchify_video order)",
                "video_bf16": ref_video.cpu(), "video_svdquant": video.cpu(),
                "video_error_fp32": (video.float() - ref_video.float()).cpu(),
                "audio_bf16": ref_audio.cpu(), "audio_svdquant": audio.cpu(),
                "audio_error_fp32": (audio.float() - ref_audio.float()).cpu(),
                "metrics": {"video": vm, "audio": am},
            }
            if any(not torch.isfinite(payload[k]).all() for k in
                   ("video_bf16", "video_svdquant", "video_error_fp32",
                    "audio_bf16", "audio_svdquant", "audio_error_fp32")):
                raise RuntimeError(f"step {step}: nonfinite saved output")
            temp = target.with_suffix(".pt.partial")
            torch.save(payload, temp)
            temp.replace(target)
            print(f"saved {target} video_nmse={vm['nmse']:.8f} audio_nmse={am['nmse']:.8f}", flush=True)
        records.append({"step": step, "path": str(target.resolve()), "sha256": sha256(target),
                        "bytes": target.stat().st_size, "metrics": payload["metrics"],
                        "unique_timesteps": payload["unique_timesteps"].tolist()})
    if any(h.act.calls == 0 for h in hooks):
        raise RuntimeError("some SVDQuant hooks were never used")
    manifest = {
        "format": "h3-e103-paired-output-manifest-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "complete-DiT output on identical BF16-trajectory input calls, quantized minus BF16",
        "prompt_id": args.expected_prompt_id, "seed": args.expected_seed,
        "height": expected["height"], "width": expected["width"], "frames": expected["frames"],
        "steps": 20, "variant": "rank32_svdquant_dynamic_activation_global_scale",
        "video_grid_thw": grid, "video_output_channels": 96, "audio_output_channels": 32,
        "calls": str(args.calls.resolve()), "calls_sha256": sha256(args.calls),
        "state": str(args.state.resolve()), "state_sha256": sha256(args.state),
        "e101_baseline": str(args.e101_report.resolve()), "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"COMPLETE {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
