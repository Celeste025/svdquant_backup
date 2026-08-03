#!/usr/bin/env python3
"""Collect per-channel token variance on Linear INPUT: raw vs postsmooth.

Same multilayer probe set as absmax Fig.1. For each forward call records
  raw_var[c]         = Var_tokens(x[..., c])
  postsmooth_var[c]  = Var_tokens(((x + shift) / s)[..., c])
then aggregates CFG branches by max (same as absmax surfaces) and stores
log2(var.clamp_min(eps)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from collect_multilayer_timestep_outliers import LAYER_SPECS, load_smooth, names_for
from collect_timestep_activation_outliers import (
    FLUX_PROMPT,
    WAN_MODEL,
    WAN_NEGATIVE,
    WAN_PROMPT,
    seed_everything,
)

VAR_EPS = 2**-40


def register_variance_hooks(
    transformer,
    raw_records: dict[str, list[torch.Tensor]],
    postsmooth_records: dict[str, list[torch.Tensor]],
    metadata: dict,
):
    modules = dict(transformer.named_modules())
    handles = []
    for key, info in metadata.items():
        module = modules[info["name"]]
        smooth = info["smooth"].float()
        shift = float(info["shift"])

        def hook(
            _module,
            inputs,
            *,
            key=key,
            smooth=smooth,
            shift=shift,
        ):
            # Move to CPU first — GPU float32 var on FLUX FFN activations OOMs
            # under model cpu_offload (same pattern as absmax postsmooth hooks).
            x = inputs[0].detach().float().cpu()
            channels = x.shape[-1]
            flat = x.reshape(-1, channels)
            raw_records[key].append(flat.var(dim=0, unbiased=False))
            xs = flat
            if shift:
                xs = xs + shift
            xs = xs / smooth.cpu().to(dtype=xs.dtype)
            postsmooth_records[key].append(xs.var(dim=0, unbiased=False))

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def stack_log2_var(
    records: list[torch.Tensor], calls_per_step: int
) -> torch.Tensor:
    values = torch.stack(records)
    expected = 50 * calls_per_step
    if values.shape[0] != expected:
        raise RuntimeError(f"expected {expected} calls, got {values.shape[0]}")
    if calls_per_step == 2:
        values = values.reshape(50, 2, -1).amax(dim=1)
    return values.clamp_min(VAR_EPS).log2()


@torch.inference_mode()
def collect(model: str, output_dir: Path) -> None:
    metadata = names_for(model)
    load_smooth(model, metadata)
    raw_records = {key: [] for key in metadata}
    postsmooth_records = {key: [] for key in metadata}
    calls_per_step = 2 if model == "wan" else 1

    if model == "wan":
        from diffusers import UniPCMultistepScheduler, WanPipeline

        pipe = WanPipeline.from_pretrained(str(WAN_MODEL), torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        handles = register_variance_hooks(
            pipe.transformer, raw_records, postsmooth_records, metadata
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
    else:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        )
        handles = register_variance_hooks(
            pipe.transformer, raw_records, postsmooth_records, metadata
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

    for handle in handles:
        handle.remove()

    surfaces_raw = {}
    surfaces_ps = {}
    for key in metadata:
        surfaces_raw[key] = stack_log2_var(raw_records[key], calls_per_step)
        surfaces_ps[key] = stack_log2_var(postsmooth_records[key], calls_per_step)
        metadata[key]["channels"] = int(surfaces_raw[key].shape[1])
        metadata[key]["shape"] = list(surfaces_raw[key].shape)

    layers = {
        k: {kk: vv for kk, vv in v.items() if kk != "smooth"}
        for k, v in metadata.items()
    }
    payload = {
        "model": model,
        "definition": (
            "log2(max_CFG Var_tokens(·)); raw = x, postsmooth = (x+shift)/s "
            f"on Linear INPUT; var floor={VAR_EPS}"
        ),
        "steps": 50,
        "seed": 44,
        "calls_per_step": calls_per_step,
        "layers": layers,
        "surfaces_raw": surfaces_raw,
        "surfaces_postsmooth": surfaces_ps,
    }
    path = output_dir / f"{model}_multilayer_variance_surfaces.pt"
    torch.save(payload, path)
    (output_dir / f"{model}_multilayer_variance_metadata.json").write_text(
        json.dumps(
            {
                "model": model,
                "path": str(path),
                "definition": payload["definition"],
                "layer_blocks": list(LAYER_SPECS[model]["blocks"]),
                "layers": layers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": model,
                "path": str(path),
                "layers": list(surfaces_raw),
                "raw_range": {
                    k: [float(v.min()), float(v.max())]
                    for k, v in surfaces_raw.items()
                },
                "postsmooth_range": {
                    k: [float(v.min()), float(v.max())]
                    for k, v in surfaces_ps.items()
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/timestep_outliers_multilayer"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collect(args.model, args.output_dir)


if __name__ == "__main__":
    main()
