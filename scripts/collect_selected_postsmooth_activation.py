#!/usr/bin/env python3
"""Collect post-SmoothQuant activation outlier and A4 error at matched layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analyze_scaled_activation_quant_error import quant_metrics
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
CAPTURE_STEPS = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49}
ROWS = 2048


def specs(model: str) -> dict[str, dict]:
    blocks = (0, 15, 29) if model == "wan" else (0, 9, 18)
    suffixes = (
        {
            "attention_q": ("attn1.to_q", None),
            "attention_out": ("attn1.to_out.0", None),
            "ffn_up": ("ffn.net.0.proj", None),
            "ffn_down": ("ffn.net.2", None),
        }
        if model == "wan"
        else {
            "attention_q": ("attn.to_q", "qkv_proj"),
            "attention_out": ("attn.to_out.0", "out_proj"),
            "ffn_up": ("ff.net.0.proj", "mlp_fc1"),
            "ffn_down": ("ff.net.2", "mlp_fc2"),
        }
    )
    prefix = "blocks" if model == "wan" else "transformer_blocks"
    result = {}
    for layer_type, (module_suffix, quant_suffix) in suffixes.items():
        for block in blocks:
            key = f"{layer_type}_b{block}"
            result[key] = {
                "module": f"{prefix}.{block}.{module_suffix}",
                "quant_key": (
                    None if quant_suffix is None else f"{prefix}.{block}.{quant_suffix}"
                ),
                "layer_type": layer_type,
                "block": block,
            }
    return result


def load_quant_params(model: str, layer_specs: dict[str, dict]) -> None:
    if model == "wan":
        payload = torch.load(WAN_CACHE, map_location="cpu", weights_only=False)
        for info in layer_specs.values():
            state = payload["state"][info["module"]]
            info["smooth"] = state["smooth"].float()
            info["shift"] = float(state["input_shift"])
            info["unsigned"] = bool(state["unsigned_activation"])
    else:
        from safetensors import safe_open

        paths = list(FLUX_CACHE.glob("*/transformer_blocks.safetensors"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one FLUX checkpoint, got {paths}")
        with safe_open(paths[0], framework="pt", device="cpu") as handle:
            for info in layer_specs.values():
                from analyze_selected_weight_reconstruction import unpack_scale

                packed = handle.get_tensor(f"{info['quant_key']}.smooth")
                info["smooth"] = unpack_scale(
                    packed, packed.numel(), 1
                ).flatten().float()
                info["shift"] = 0.171875 if info["layer_type"] == "ffn_down" else 0.0
                info["unsigned"] = info["layer_type"] == "ffn_down"


def register(transformer, model: str, layer_specs: dict[str, dict], reports: list[dict]):
    modules = dict(transformer.named_modules())
    handles = []
    calls = {key: 0 for key in layer_specs}
    calls_per_step = 2 if model == "wan" else 1

    for key, info in layer_specs.items():
        def hook(_module, inputs, *, key=key, info=info):
            call = calls[key]
            calls[key] += 1
            step = call // calls_per_step
            branch = call % calls_per_step
            if step not in CAPTURE_STEPS:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(ROWS, x.shape[0])
            index = torch.linspace(0, x.shape[0] - 1, count, device=x.device).long()
            sample = x.index_select(0, index).detach().float().cpu()
            if info["shift"]:
                sample.add_(info["shift"])
            result = quant_metrics(
                sample, info["smooth"].float().cpu(), info["unsigned"]
            )
            result.update(
                {
                    "model": model,
                    "key": key,
                    "module": info["module"],
                    "layer_type": info["layer_type"],
                    "block": info["block"],
                    "step": step + 1,
                    "cfg_branch": branch if calls_per_step == 2 else None,
                    "sampled_tokens": count,
                    "total_tokens": x.shape[0],
                }
            )
            reports.append(result)

        handles.append(modules[info["module"]].register_forward_pre_hook(hook))
    return handles


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quant_error_diagnosis/postsmooth_activation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_specs = specs(args.model)
    load_quant_params(args.model, layer_specs)
    reports = []

    if args.model == "wan":
        from diffusers import UniPCMultistepScheduler, WanPipeline

        pipe = WanPipeline.from_pretrained(str(WAN_MODEL), torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        handles = register(pipe.transformer, args.model, layer_specs, reports)
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
        handles = register(pipe.transformer, args.model, layer_specs, reports)
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
    path = args.output_dir / f"{args.model}_metrics.json"
    path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"saved {len(reports)} records to {path}")


if __name__ == "__main__":
    main()
