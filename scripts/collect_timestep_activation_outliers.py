#!/usr/bin/env python3
"""Reproduce SVQ-GPTQ Fig.1-style per-channel timestep activation surfaces."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


WAN_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)
WAN_PROMPT = "An astronaut feeding ducks on a sunny afternoon, reflection from the water."
WAN_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
FLUX_PROMPT = "A cat holding a sign that says hello world"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def register_absmax_hook(module: torch.nn.Module, records: list[torch.Tensor]):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        channels = tensor.shape[-1]
        absmax = tensor.detach().float().abs().reshape(-1, channels).amax(dim=0)
        records.append(absmax.cpu())

    return module.register_forward_hook(hook)


def collect_wan(output: Path) -> dict:
    from diffusers import UniPCMultistepScheduler, WanPipeline

    pipe = WanPipeline.from_pretrained(str(WAN_MODEL), torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    layer_name = "blocks.20.ffn.net.2"
    layer = dict(pipe.transformer.named_modules())[layer_name]
    records: list[torch.Tensor] = []
    handle = register_absmax_hook(layer, records)
    pipe.enable_model_cpu_offload()
    seed_everything(44)
    pipe(
        prompt=WAN_PROMPT,
        negative_prompt=WAN_NEGATIVE,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=50,
        guidance_scale=6.0,
        generator=torch.Generator(device="cuda").manual_seed(44),
        output_type="latent",
    )
    handle.remove()
    values = torch.stack(records)
    # Wan CFG invokes the transformer twice per denoising step. Preserve both
    # branches in the raw file and aggregate their maxima for the Fig.1 surface.
    if values.shape[0] != 100:
        raise RuntimeError(f"expected 100 Wan calls, got {values.shape[0]}")
    surface = values.reshape(50, 2, -1).amax(dim=1).clamp_min_(2**-20).log2()
    torch.save({"raw_absmax": values, "log2_absmax": surface}, output)
    return {
        "model": "Wan2.1-T2V-1.3B",
        "layer": layer_name,
        "seed": 44,
        "steps": 50,
        "cfg_branches_per_step": 2,
        "channels": surface.shape[1],
        "surface_path": str(output),
    }


def collect_flux(output: Path) -> dict:
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    layer_name = "transformer_blocks.12.ff.net.2"
    layer = dict(pipe.transformer.named_modules())[layer_name]
    records: list[torch.Tensor] = []
    handle = register_absmax_hook(layer, records)
    pipe.enable_model_cpu_offload(gpu_id=0)
    seed_everything(44)
    pipe(
        FLUX_PROMPT,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        generator=torch.Generator(device="cuda").manual_seed(44),
        output_type="latent",
    )
    handle.remove()
    values = torch.stack(records)
    if values.shape[0] != 50:
        raise RuntimeError(f"expected 50 FLUX calls, got {values.shape[0]}")
    surface = values.clamp_min_(2**-20).log2()
    torch.save({"raw_absmax": values, "log2_absmax": surface}, output)
    return {
        "model": "FLUX.1-dev",
        "layer": layer_name,
        "seed": 44,
        "steps": 50,
        "cfg_branches_per_step": 1,
        "channels": surface.shape[1],
        "surface_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/timestep_outliers"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = args.output_dir / f"{args.model}_surface.pt"
    metadata = collect_wan(surface_path) if args.model == "wan" else collect_flux(surface_path)
    (args.output_dir / f"{args.model}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
