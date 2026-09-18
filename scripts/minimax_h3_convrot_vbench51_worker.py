#!/usr/bin/env python3
"""Resumable ConvRot generator for the fixed MiniMax-H3 VBench-51 contract."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path(os.environ.get("DIFFSYNTH_ROOT", ROOT / "third_party" / "DiffSynth-Studio"))
if not DIFFSYNTH_ROOT.is_dir():
    raise FileNotFoundError(f"DiffSynth-Studio not found: {DIFFSYNTH_ROOT}")
sys.path.insert(0, str(DIFFSYNTH_ROOT))
os.chdir(DIFFSYNTH_ROOT)

import torch
from diffsynth.utils.data.audio_video import write_video_audio
from minimax_h3_convrot_common import assert_target_structure, configure_offloaded_quant_linear, import_convrot, state_config, unpack_weight, warmup_convrot_extension
from minimax_h3_svdquant_common import load_h3_pipeline

SETTINGS = {"seed": 0, "height": 576, "width": 1024, "frames": 124, "steps": 20,
            "cfg_scale": 1.0, "rand_device": "cpu", "tiled": True, "fps": 24}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def complete(video: Path, status: Path) -> bool:
    try:
        return video.is_file() and json.loads(status.read_text()).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


@torch.no_grad()
def apply_state(dit, state_path: Path) -> dict:
    import_convrot(); warmup_convrot_extension()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-convrot-nvfp4-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("invalid H3 ConvRot state")
    from convrot.rotated_nvfp4_tensor import RotatedNVFP4Tensor
    targets = assert_target_structure(dit)
    for index, (name, linear) in enumerate(targets, 1):
        linear.weight = torch.nn.Parameter(unpack_weight(state["layers"][name], "cuda"), requires_grad=False)
        if not isinstance(linear.weight, RotatedNVFP4Tensor):
            raise RuntimeError(f"{name}: state did not restore RotatedNVFP4Tensor")
        configure_offloaded_quant_linear(linear)
        if index % 20 == 0:
            print(f"ConvRot restored {index}/200", flush=True)
    return state


def generate(pipe, case: dict, video: Path) -> None:
    frames, audio = pipe(prompt=case["prompt"], seed=0, height=576, width=1024, num_frames=124,
                         num_inference_steps=20, cfg_scale=1.0, rand_device="cpu", tiled=True)
    if len(frames) != 124:
        raise RuntimeError(f"expected 124 frames, got {len(frames)}")
    video.parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(video=frames, audio=audio, output_path=str(video), fps=24,
                      audio_sample_rate=pipe.audio_vae.sample_rate)
    del frames, audio
    gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("settings") != SETTINGS or len(manifest.get("cases", [])) != 51:
        raise RuntimeError("manifest does not match H3 VBench-51 contract")
    cases = manifest["cases"][:1] if args.smoke else manifest["cases"]
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    state = apply_state(pipe.dit, args.state)
    status_path = args.output / "worker_status" / "convrot.json"
    atomic_json(status_path, {"state": "running", "pid": os.getpid(), "quant_state": str(args.state),
                              "config": state_config(), "targets": 200, "started_at": time.time()})
    generated = skipped = 0; failures = []
    for case in cases:
        d = args.output / "cases" / case["case_id"]
        video, status = d / "convrot.mp4", d / "convrot.json"
        if complete(video, status):
            skipped += 1; continue
        started = time.time()
        try:
            generate(pipe, case, video)
            atomic_json(status, {"state": "complete", "variant": "convrot", "case_id": case["case_id"],
                                 "prompt": case["prompt"], "dimensions": case["dimensions"], **SETTINGS,
                                 "quant_state": str(args.state), "video": str(video), "seconds": time.time()-started})
            generated += 1
        except Exception as exc:
            failures.append({"case_id": case["case_id"], "error": repr(exc)})
            atomic_json(status, {"state": "failed", "variant": "convrot", "case_id": case["case_id"], "error": repr(exc)})
            torch.cuda.empty_cache()
    atomic_json(status_path, {"state": "complete" if not failures else "completed_with_failures", "quant_state": str(args.state),
                              "config": state_config(), "targets": 200, "generated": generated, "skipped": skipped,
                              "failures": failures, "finished_at": time.time()})
    if failures:
        raise SystemExit(f"ConvRot failed cases: {len(failures)}")


if __name__ == "__main__":
    main()
