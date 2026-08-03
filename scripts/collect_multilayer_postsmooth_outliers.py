#!/usr/bin/env python3
"""Collect postsmooth Linear-INPUT absmax surfaces for multilayer probe layers.

Same layer set as collect_multilayer_timestep_outliers.py, but records
  postsmooth = log2(absmax((x + shift) / smooth))
via forward_pre_hooks (SmoothQuant acts on Linear inputs).
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


def register_postsmooth_hooks(
    transformer,
    records: dict[str, list[torch.Tensor]],
    metadata: dict,
):
    modules = dict(transformer.named_modules())
    handles = []
    for key, info in metadata.items():
        module = modules[info["name"]]
        smooth = info["smooth"].float()
        shift = float(info["shift"])

        def hook(_module, inputs, *, key=key, smooth=smooth, shift=shift):
            x = inputs[0].detach().float().cpu()
            channels = x.shape[-1]
            flat = x.reshape(-1, channels)
            if shift:
                flat = flat + shift
            xs = flat / smooth.cpu().to(dtype=flat.dtype)
            records[key].append(xs.abs().amax(dim=0))

        handles.append(module.register_forward_pre_hook(hook))
    return handles


@torch.inference_mode()
def collect(model: str, output_dir: Path) -> None:
    metadata = names_for(model)
    load_smooth(model, metadata)
    records = {key: [] for key in metadata}
    calls_per_step = 2 if model == "wan" else 1

    if model == "wan":
        from diffusers import UniPCMultistepScheduler, WanPipeline

        pipe = WanPipeline.from_pretrained(str(WAN_MODEL), torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        handles = register_postsmooth_hooks(pipe.transformer, records, metadata)
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
        handles = register_postsmooth_hooks(pipe.transformer, records, metadata)
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

    surfaces = {}
    for key, values in records.items():
        raw = torch.stack(values)
        expected = 50 * calls_per_step
        if raw.shape[0] != expected:
            raise RuntimeError(f"{key}: expected {expected} calls, got {raw.shape[0]}")
        if calls_per_step == 2:
            raw = raw.reshape(50, 2, -1).amax(dim=1)
        surfaces[key] = raw.clamp_min(2**-20).log2()
        metadata[key]["channels"] = int(raw.shape[1])
        metadata[key]["shape"] = list(surfaces[key].shape)

    payload = {
        "model": model,
        "definition": (
            "log2(max over batch,tokens[,CFG branches] of abs((x+shift)/smooth)) "
            "on Linear INPUT after SmoothQuant"
        ),
        "steps": 50,
        "seed": 44,
        "calls_per_step": calls_per_step,
        "layers": {
            k: {kk: vv for kk, vv in v.items() if kk != "smooth"}
            for k, v in metadata.items()
        },
        "surfaces": surfaces,
    }
    path = output_dir / f"{model}_multilayer_postsmooth_surfaces.pt"
    torch.save(payload, path)
    (output_dir / f"{model}_multilayer_postsmooth_metadata.json").write_text(
        json.dumps(
            {
                "model": model,
                "path": str(path),
                "definition": payload["definition"],
                "layer_blocks": list(LAYER_SPECS[model]["blocks"]),
                "layers": payload["layers"],
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
                "layers": list(surfaces),
                "channels": {k: int(v.shape[1]) for k, v in surfaces.items()},
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
