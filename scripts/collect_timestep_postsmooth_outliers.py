#!/usr/bin/env python3
"""Collect raw vs postsmooth Linear-input absmax surfaces (same layers as Fig.1).

Original collect_timestep_activation_outliers.py hooks layer *outputs*.
SmoothQuant acts on Linear *inputs*, so this script uses forward_pre_hooks and
records both:
  raw:       absmax(x) per channel
  postsmooth: absmax((x + shift) / smooth) per channel
for Wan blocks.20.ffn.net.2 and FLUX transformer_blocks.12.ff.net.2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from collect_timestep_activation_outliers import (
    FLUX_PROMPT,
    WAN_MODEL,
    WAN_NEGATIVE,
    WAN_PROMPT,
    seed_everything,
)


WAN_CACHE = Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt")
FLUX_CACHE = Path(
    "/data/home/jinqiwen/.cache/huggingface/hub/"
    "models--mit-han-lab--svdq-int4-flux.1-dev/snapshots"
)
LAYERS = {
    "wan": "blocks.20.ffn.net.2",
    "flux": "transformer_blocks.12.ff.net.2",
}
FLUX_QUANT_KEY = "transformer_blocks.12.mlp_fc2"


def load_smooth(model: str) -> tuple[torch.Tensor, float]:
    if model == "wan":
        payload = torch.load(WAN_CACHE, map_location="cpu", weights_only=False)
        state = payload["state"][LAYERS["wan"]]
        return state["smooth"].float(), float(state["input_shift"])
    from safetensors import safe_open

    from analyze_selected_weight_reconstruction import unpack_scale

    paths = list(FLUX_CACHE.glob("*/transformer_blocks.safetensors"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one FLUX checkpoint, got {paths}")
    with safe_open(paths[0], framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(f"{FLUX_QUANT_KEY}.smooth")
        smooth = unpack_scale(packed, packed.numel(), 1).flatten().float()
    # FFN-down: unsigned path uses the same shift as SVDQuant / postsmooth scripts
    return smooth, 0.171875


def register_input_hooks(
    module: torch.nn.Module,
    *,
    smooth: torch.Tensor,
    shift: float,
    raw_records: list[torch.Tensor],
    smooth_records: list[torch.Tensor],
):
    def hook(_module, inputs):
        # Move to CPU immediately — FLUX activations are large and the DiT
        # already fills most of the GPU under cpu_offload.
        x = inputs[0].detach().float().cpu()
        channels = x.shape[-1]
        flat = x.reshape(-1, channels)
        raw_records.append(flat.abs().amax(dim=0))
        xs = flat
        if shift:
            xs = xs + shift
        xs = xs / smooth.cpu().to(dtype=xs.dtype)
        smooth_records.append(xs.abs().amax(dim=0))

    return module.register_forward_pre_hook(hook)


def stack_surface(records: list[torch.Tensor], calls_per_step: int) -> torch.Tensor:
    values = torch.stack(records)
    expected = 50 * calls_per_step
    if values.shape[0] != expected:
        raise RuntimeError(f"expected {expected} calls, got {values.shape[0]}")
    if calls_per_step == 2:
        values = values.reshape(50, 2, -1).amax(dim=1)
    return values.clamp_min(2**-20).log2()


@torch.inference_mode()
def collect_wan(output_dir: Path) -> dict:
    from diffusers import UniPCMultistepScheduler, WanPipeline

    smooth, shift = load_smooth("wan")
    pipe = WanPipeline.from_pretrained(str(WAN_MODEL), torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=3.0
    )
    layer = dict(pipe.transformer.named_modules())[LAYERS["wan"]]
    raw_records: list[torch.Tensor] = []
    smooth_records: list[torch.Tensor] = []
    handle = register_input_hooks(
        layer,
        smooth=smooth,
        shift=shift,
        raw_records=raw_records,
        smooth_records=smooth_records,
    )
    pipe.enable_model_cpu_offload(gpu_id=0)
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
    raw_surface = stack_surface(raw_records, 2)
    smooth_surface = stack_surface(smooth_records, 2)
    torch.save(
        {
            "raw_log2_absmax": raw_surface,
            "postsmooth_log2_absmax": smooth_surface,
            "shift": shift,
            "smooth": smooth.cpu(),
            "layer": LAYERS["wan"],
            "definition": "log2(per-channel absmax of Linear INPUT; postsmooth=(x+shift)/s)",
        },
        output_dir / "wan_input_raw_vs_postsmooth.pt",
    )
    return {
        "model": "Wan2.1-1.3B",
        "layer": LAYERS["wan"],
        "shift": shift,
        "channels": int(raw_surface.shape[1]),
        "cfg_branches_per_step": 2,
    }


@torch.inference_mode()
def collect_flux(output_dir: Path) -> dict:
    from diffusers import FluxPipeline

    smooth, shift = load_smooth("flux")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    layer = dict(pipe.transformer.named_modules())[LAYERS["flux"]]
    raw_records: list[torch.Tensor] = []
    smooth_records: list[torch.Tensor] = []
    handle = register_input_hooks(
        layer,
        smooth=smooth,
        shift=shift,
        raw_records=raw_records,
        smooth_records=smooth_records,
    )
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
    raw_surface = stack_surface(raw_records, 1)
    smooth_surface = stack_surface(smooth_records, 1)
    torch.save(
        {
            "raw_log2_absmax": raw_surface,
            "postsmooth_log2_absmax": smooth_surface,
            "shift": shift,
            "smooth": smooth.cpu(),
            "layer": LAYERS["flux"],
            "definition": "log2(per-channel absmax of Linear INPUT; postsmooth=(x+shift)/s)",
        },
        output_dir / "flux_input_raw_vs_postsmooth.pt",
    )
    return {
        "model": "FLUX.1-dev",
        "layer": LAYERS["flux"],
        "shift": shift,
        "channels": int(raw_surface.shape[1]),
        "cfg_branches_per_step": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux"), required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/timestep_outliers")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_wan(args.output_dir) if args.model == "wan" else collect_flux(args.output_dir)
    (args.output_dir / f"{args.model}_input_raw_vs_postsmooth_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
