#!/usr/bin/env python3
"""Collect post-smoothing activation INT4 metrics from BF16 FLUX.1-dev.

Fixes vs earlier version:
- unpack packed Nunchaku smooth scales
- FFN-down uses shift + unsigned INT4 (matches SVDQuant / matched postsmooth)
- prompt/seed aligned with other diagnostic scripts
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import FluxPipeline
from safetensors import safe_open

from analyze_selected_weight_reconstruction import unpack_scale
from collect_timestep_activation_outliers import FLUX_PROMPT, seed_everything
from quant_error_metrics import quant_metrics


BASE_MODEL = "black-forest-labs/FLUX.1-dev"
QUANT_CACHE = Path(
    "/data/home/jinqiwen/.cache/huggingface/hub/"
    "models--mit-han-lab--svdq-int4-flux.1-dev/snapshots"
)
BLOCKS = (0, 6, 12, 18)
ROWS_PER_CALL = 64


def checkpoint_path() -> Path:
    snapshots = list(QUANT_CACHE.glob("*/transformer_blocks.safetensors"))
    if len(snapshots) != 1:
        raise RuntimeError(f"expected one quant snapshot, found {snapshots}")
    return snapshots[0]


def mappings() -> list[tuple[str, str, str]]:
    """(module_name, smooth_key, layer_kind)."""
    local = {
        "qkv_proj": ("attn.to_q", "attn"),
        "qkv_proj_context": ("attn.add_q_proj", "attn"),
        "out_proj": ("attn.to_out.0", "attn"),
        "out_proj_context": ("attn.to_add_out", "attn"),
        "mlp_fc1": ("ff.net.0.proj", "ffn_up"),
        "mlp_fc2": ("ff.net.2", "ffn_down"),
        "mlp_context_fc1": ("ff_context.net.0.proj", "ffn_up"),
        "mlp_context_fc2": ("ff_context.net.2", "ffn_down"),
    }
    result = []
    for block in BLOCKS:
        for quant_name, (module_name, kind) in local.items():
            result.append(
                (
                    f"transformer_blocks.{block}.{module_name}",
                    f"transformer_blocks.{block}.{quant_name}.smooth",
                    kind,
                )
            )
    return result


def main() -> None:
    output_dir = Path("outputs/activation_error_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    smooths: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    with safe_open(checkpoint_path(), framework="pt", device="cpu") as handle:
        for module_name, key, kind in mappings():
            packed = handle.get_tensor(key)
            smooth = unpack_scale(packed, packed.numel(), 1).flatten().float()
            smooths[module_name] = smooth
            meta[module_name] = {
                "kind": kind,
                "shift": 0.171875 if kind == "ffn_down" else 0.0,
                "unsigned": kind == "ffn_down",
            }

    pipeline = FluxPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload(gpu_id=0)
    modules = dict(pipeline.transformer.named_modules())
    reports: list[dict] = []
    handles = []

    def hook(name: str):
        smooth_cpu = smooths[name]
        info = meta[name]

        def collect(_module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(ROWS_PER_CALL, x.shape[0])
            indices = torch.linspace(0, x.shape[0] - 1, count, device=x.device).long()
            sample = x.index_select(0, indices).detach().float().cpu()
            if info["shift"]:
                sample = sample + info["shift"]
            metrics = quant_metrics(
                sample, smooth_cpu.float().cpu(), unsigned=info["unsigned"]
            )
            metrics.update(
                {
                    "model": "FLUX.1-dev",
                    "name": name,
                    "kind": info["kind"],
                    "shift": info["shift"],
                    "unsigned": info["unsigned"],
                    "call_index": sum(r["name"] == name for r in reports),
                    "rows": count,
                }
            )
            reports.append(metrics)

        return collect

    for name in smooths:
        handles.append(modules[name].register_forward_pre_hook(hook(name)))

    seed_everything(44)
    generator = torch.Generator(device="cuda:0").manual_seed(44)
    with torch.inference_mode():
        image = pipeline(
            FLUX_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=3.5,
            generator=generator,
        ).images[0]
    image.save(output_dir / "flux_dev_bf16_probe_seed44.png")
    for handle in handles:
        handle.remove()
    (output_dir / "flux_metrics.json").write_text(json.dumps(reports, indent=2))
    print(f"saved {len(reports)} module-call records")


if __name__ == "__main__":
    main()
