#!/usr/bin/env python3
"""Persistent, resumable MiniMax-H3 generator for the VBench-51 subset."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path("/home/wjq/workspace/DiffSynth-Studio")
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
os.chdir(DIFFSYNTH_ROOT)  # DiffSynth model patterns are relative to this directory.

import torch
from diffsynth.utils.data.audio_video import write_video_audio

from minimax_h3_svdquant_common import (
    LowRankBranch,
    install_runtime_hooks,
    load_h3_pipeline,
    nvfp4_qdq,
    target_linears,
)

STATE = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt"
SETTINGS = {
    "seed": 0, "height": 576, "width": 1024, "frames": 124, "steps": 20,
    "cfg_scale": 1.0, "rand_device": "cpu", "tiled": True, "fps": 24,
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def complete(video: Path, status: Path) -> bool:
    if not video.is_file() or not status.is_file():
        return False
    try:
        return json.loads(status.read_text()).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


@torch.no_grad()
def install_quantized(linear, qweight, smooth=None, a=None, b=None):
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
def apply_plain_w4a4(dit):
    runtimes = []
    for index, (_, linear) in enumerate(target_linears(dit), 1):
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        runtimes.append(install_quantized(linear, nvfp4_qdq(weight)))
        del weight
        if index % 20 == 0:
            print(f"plain W4A4 installed: {index}/200", flush=True)
        torch.cuda.empty_cache()
    return runtimes


@torch.no_grad()
def apply_svdquant(dit, state_path: Path):
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("invalid MiniMax-H3 standard SVDQuant state")
    cfg = state.get("config", {})
    expected = {"rank": 32, "num_grids": 10, "max_lowrank_iters": 50, "group_size": 16, "element_size": 128}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"unexpected SVDQuant recipe: {cfg}")
    runtimes = []
    for index, (name, linear) in enumerate(target_linears(dit), 1):
        layer = state["layers"][name]
        weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
        smooth = layer["smooth"].to(weight)
        a, b = layer["final_a"].to(weight), layer["final_b"].to(weight)
        runtimes.append(install_quantized(linear, nvfp4_qdq(weight * smooth - b @ a), smooth, a, b))
        del weight, smooth, a, b
        if index % 20 == 0:
            print(f"SVDQuant installed: {index}/200", flush=True)
        torch.cuda.empty_cache()
    return runtimes


def validate(dit, variant: str, runtimes) -> dict[str, int]:
    targets = target_linears(dit)
    # LowRankBranch is deliberately retained by each runtime hook closure rather
    # than registered under the DiT; registering it would alter checkpoint/module
    # traversal. Validate the hooks' owned branches, not ``dit.modules()``.
    branches = sum(runtime.branch is not None for runtime in runtimes)
    if variant == "w4a4" and branches:
        raise RuntimeError(f"plain W4A4 unexpectedly has {branches} low-rank branches")
    if variant == "svdquant" and branches != 200:
        raise RuntimeError(f"SVDQuant expected 200 low-rank branches, found {branches}")
    if variant != "bf16" and len(runtimes) != 200:
        raise RuntimeError(f"{variant} expected 200 quant runtimes, found {len(runtimes)}")
    return {"target_linears": len(targets), "lowrank_branches": branches, "activation_quantizers": len(runtimes)}


def generate(pipe, case: dict, video: Path) -> None:
    output, audio = pipe(
        prompt=case["prompt"], seed=SETTINGS["seed"], height=SETTINGS["height"], width=SETTINGS["width"],
        num_frames=SETTINGS["frames"], num_inference_steps=SETTINGS["steps"], cfg_scale=SETTINGS["cfg_scale"],
        rand_device=SETTINGS["rand_device"], tiled=SETTINGS["tiled"],
    )
    if len(output) != SETTINGS["frames"]:
        raise RuntimeError(f"expected {SETTINGS['frames']} frames, got {len(output)}")
    video.parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(video=output, audio=audio, output_path=str(video), fps=SETTINGS["fps"],
                      audio_sample_rate=pipe.audio_vae.sample_rate)
    del output, audio
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("bf16", "w4a4", "svdquant"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("settings") != SETTINGS:
        raise RuntimeError("manifest settings differ from the H3 calibration contract")
    cases = manifest["cases"][:1] if args.smoke else manifest["cases"]
    if len(manifest["cases"]) != 51:
        raise RuntimeError("expected the deterministic 51-case VBench subset")

    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    runtimes = []
    if args.variant != "bf16":
        pipe.load_models_to_device(["dit"])
        runtimes = apply_plain_w4a4(pipe.dit) if args.variant == "w4a4" else apply_svdquant(pipe.dit, STATE)
    validation = validate(pipe.dit, args.variant, runtimes)
    worker_status = args.output / "worker_status" / f"{args.variant}.json"
    atomic_json(worker_status, {"state": "running", "variant": args.variant, "validation": validation,
                                "pid": os.getpid(), "started_at": time.time()})
    generated = skipped = 0
    failures = []
    for case in cases:
        case_dir = args.output / "cases" / case["case_id"]
        video, status = case_dir / f"{args.variant}.mp4", case_dir / f"{args.variant}.json"
        if complete(video, status):
            skipped += 1
            continue
        started = time.time()
        try:
            generate(pipe, case, video)
            if args.variant != "bf16" and any(runtime.act.calls == 0 for runtime in runtimes):
                raise RuntimeError("a quantized activation hook was not called")
            atomic_json(status, {"state": "complete", "variant": args.variant, "case_id": case["case_id"],
                "prompt": case["prompt"], "dimensions": case["dimensions"], **SETTINGS,
                "video": str(video), "seconds": time.time() - started})
            generated += 1
        except Exception as exc:
            failures.append({"case_id": case["case_id"], "error": repr(exc)})
            atomic_json(status, {"state": "failed", "variant": args.variant, "case_id": case["case_id"],
                                 "error": repr(exc)})
            torch.cuda.empty_cache()
    atomic_json(worker_status, {"state": "complete" if not failures else "completed_with_failures",
                                "variant": args.variant, "validation": validation, "generated": generated,
                                "skipped": skipped, "failures": failures, "finished_at": time.time()})
    if failures:
        raise SystemExit(f"{args.variant}: {len(failures)} case(s) failed")


if __name__ == "__main__":
    main()
