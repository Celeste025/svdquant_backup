#!/usr/bin/env python3
"""Generate rank-64 H3 SVDQuant videos with all blocks.0 linears restored to BF16.

Every target linear outside block 0 keeps the exact standard SVDQuant runtime:
smooth-folded NVFP4 weights, per-layer activation quantization, and the low-rank
branch.  The four block-0 linears are swapped back to the pristine BF16
checkpoint weights with no hooks at all, so no smoothing scale or branch can
touch them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = Path(os.environ.get("DIFFSYNTH_ROOT", ROOT / "third_party/DiffSynth-Studio"))
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
os.chdir(DIFFSYNTH_ROOT)

import torch
from diffsynth.utils.data.audio_video import write_video_audio

from minimax_h3_svdquant_common import DATA_ROOT, install_runtime_hooks, load_h3_pipeline, nvfp4_qdq, target_linears

DEFAULT_STATE = DATA_ROOT / "artifacts/variants/nvfp4-g10-r64/quant_state.pt"
DEFAULT_MANIFEST = DATA_ROOT / "videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench51_r64/manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "runs/minimax-h3/samples/nvfp4-g10-r64-block0bf16"
RESTORE_BLOCK = 0
SETTINGS = {"seed": 0, "height": 576, "width": 1024, "frames": 124, "steps": 20,
            "cfg_scale": 1.0, "rand_device": "cpu", "tiled": True, "fps": 24}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cases", nargs="+", default=["vbench_001", "vbench_166"])
    p.add_argument("--vram-limit-gib", type=float, default=None,
                   help="device-wide VRAM ceiling for DiffSynth offload")
    p.add_argument("--reserve-gib", type=float, default=6.0)
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
def restore_bf16(linear) -> None:
    """Put the pristine BF16 checkpoint weight back and leave the layer hook-free."""
    weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
    linear.load_state_dict({"weight": weight.detach().cpu()}, assign=True, strict=False)
    linear.disk_offload = False
    linear.offload_dtype = linear.onload_dtype = torch.bfloat16
    linear.offload_device = linear.onload_device = torch.device("cpu")
    linear.computation_dtype = torch.bfloat16
    linear.computation_device = torch.device("cuda")
    linear.vram_limit = 0.0
    linear.state = 1
    del weight


@torch.no_grad()
def apply(dit, state_path: Path):
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(state.get("layers", {})) != 200:
        raise RuntimeError("invalid H3 SVDQuant state")
    cfg = state["config"]
    expected = {"rank": 64, "num_grids": 10, "max_lowrank_iters": 50, "group_size": 16, "element_size": 128}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"unexpected SVDQuant recipe: {cfg}")
    restored, runtimes = [], {}
    for index, (name, linear) in enumerate(target_linears(dit), 1):
        item = state["layers"][name]
        if name.startswith(f"blocks.{RESTORE_BLOCK}."):
            restore_bf16(linear)
            restored.append(name)
        else:
            weight, _ = linear.load_from_disk(torch.bfloat16, "cuda", assign=False)
            smooth, a, b = item["smooth"].to(weight), item["final_a"].to(weight), item["final_b"].to(weight)
            runtimes[name] = install_svdq(linear, nvfp4_qdq(weight * smooth - b @ a), item["smooth"], item["final_a"], item["final_b"])
            del weight, smooth, a, b
        if index % 20 == 0:
            print(f"prepared {index}/200", flush=True)
        torch.cuda.empty_cache()
    return restored, runtimes


def validate(dit, restored: list[str], runtimes: dict) -> dict:
    if len(restored) != 4 or any(not name.startswith(f"blocks.{RESTORE_BLOCK}.") for name in restored):
        raise RuntimeError(f"expected exactly the 4 block-{RESTORE_BLOCK} linears restored, got {restored}")
    if len(runtimes) != 196 or any(runtime.branch is None for runtime in runtimes.values()):
        raise RuntimeError(f"expected 196 hooked SVDQuant layers with branches, got {len(runtimes)}")
    for name in restored:
        linear = dict(target_linears(dit))[name]
        if linear._forward_hooks or linear._forward_pre_hooks:
            raise RuntimeError(f"{name} still carries hooks after BF16 restore")
        if linear.weight.dtype != torch.bfloat16:
            raise RuntimeError(f"{name} weight dtype is {linear.weight.dtype}, not bfloat16")
        pristine, _ = linear.load_from_disk(torch.bfloat16, "cpu", assign=False)
        if not torch.equal(linear.weight.data, pristine.to(torch.bfloat16)):
            raise RuntimeError(f"{name} weight differs from the pristine BF16 checkpoint")
        del pristine
    return {"restored_bf16_layers": sorted(restored), "svdquant_layers": 196,
            "lowrank_branches": 196, "activation_quantizers": 196}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("settings") != SETTINGS:
        raise RuntimeError("manifest settings differ from the H3 vbench51 contract")
    if manifest["variants"]["svdquant"].get("rank") != 64:
        raise RuntimeError("manifest does not describe the rank-64 variant")
    wanted = {case["case_id"]: case for case in manifest["cases"]}
    missing = [c for c in args.cases if c not in wanted]
    if missing:
        raise KeyError(f"cases not in manifest: {missing}")

    pipe = load_h3_pipeline(full=True, reserve_gib=args.reserve_gib, vram_limit_gib=args.vram_limit_gib)
    pipe.load_models_to_device(["dit"])
    restored, runtimes = apply(pipe.dit, args.state)
    validation = validate(pipe.dit, restored, runtimes)
    print(f"validation: {json.dumps({k: v for k, v in validation.items() if k != 'restored_bf16_layers'})}", flush=True)

    for case_id in args.cases:
        case = wanted[case_id]
        target = args.output_dir / f"svdquant_block0bf16_{case_id}_seed{SETTINGS['seed']}_{SETTINGS['height']}x{SETTINGS['width']}_{SETTINGS['frames']}f_{SETTINGS['steps']}steps.mp4"
        sidecar = target.with_suffix(".json")
        if target.is_file() and sidecar.is_file() and json.loads(sidecar.read_text()).get("state") == "complete":
            print(f"skip existing {target}", flush=True)
            continue
        started = time.time()
        video, audio = pipe(prompt=case["prompt"], seed=SETTINGS["seed"], height=SETTINGS["height"], width=SETTINGS["width"],
                            num_frames=SETTINGS["frames"], num_inference_steps=SETTINGS["steps"], cfg_scale=SETTINGS["cfg_scale"],
                            rand_device=SETTINGS["rand_device"], tiled=SETTINGS["tiled"])
        if len(video) != SETTINGS["frames"]:
            raise RuntimeError(f"{case_id}: expected {SETTINGS['frames']} frames, got {len(video)}")
        if any(runtime.act.calls == 0 for runtime in runtimes.values()):
            raise RuntimeError(f"{case_id}: a quantized activation hook was never called")
        write_video_audio(video=video, audio=audio, output_path=str(target), fps=SETTINGS["fps"],
                          audio_sample_rate=pipe.audio_vae.sample_rate)
        sidecar.write_text(json.dumps({"state": "complete", "variant": "svdquant-block0bf16", "case_id": case_id,
                                       "prompt": case["prompt"], "quant_state": str(args.state), **SETTINGS,
                                       "restored_layers": validation["restored_bf16_layers"],
                                       "video": str(target), "seconds": time.time() - started}, indent=2, ensure_ascii=False))
        print(f"saved {target} ({time.time() - started:.0f}s)", flush=True)
        del video, audio
        torch.cuda.empty_cache()
    print("ALL CASES DONE", flush=True)


if __name__ == "__main__":
    main()
