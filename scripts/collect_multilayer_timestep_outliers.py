#!/usr/bin/env python3
"""Collect Fig.1-style timestep/channel absmax surfaces AND postsmooth A4 MSE/NMSE."""

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
from quant_error_metrics import quant_metrics


LAYER_SPECS = {
    "wan": {
        "blocks": (0, 15, 29),
        "types": {
            "attention_q": "attn1.to_q",
            "attention_out": "attn1.to_out.0",
            "ffn_up": "ffn.net.0.proj",
            "ffn_down": "ffn.net.2",
        },
    },
    "flux": {
        "blocks": (0, 9, 18),
        "types": {
            "attention_q": "attn.to_q",
            "attention_out": "attn.to_out.0",
            "ffn_up": "ff.net.0.proj",
            "ffn_down": "ff.net.2",
        },
    },
}

WAN_CACHE = Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt")
FLUX_CACHE = Path(
    "/data/home/jinqiwen/.cache/huggingface/hub/"
    "models--mit-han-lab--svdq-int4-flux.1-dev/snapshots"
)
A4_ROWS = 2048


def names_for(model: str) -> dict[str, dict]:
    prefix = "blocks" if model == "wan" else "transformer_blocks"
    spec = LAYER_SPECS[model]
    flux_quant = {
        "attention_q": "qkv_proj",
        "attention_out": "out_proj",
        "ffn_up": "mlp_fc1",
        "ffn_down": "mlp_fc2",
    }
    result = {}
    for layer_type, suffix in spec["types"].items():
        for block in spec["blocks"]:
            key = f"{layer_type}_b{block}"
            result[key] = {
                "name": f"{prefix}.{block}.{suffix}",
                "type": layer_type,
                "block": block,
                "quant_key": (
                    None
                    if model == "wan"
                    else f"{prefix}.{block}.{flux_quant[layer_type]}"
                ),
            }
    return result


def load_smooth(model: str, metadata: dict[str, dict]) -> None:
    if model == "wan":
        payload = torch.load(WAN_CACHE, map_location="cpu", weights_only=False)
        for info in metadata.values():
            state = payload["state"][info["name"]]
            info["smooth"] = state["smooth"].float()
            info["shift"] = float(state["input_shift"])
            info["unsigned"] = bool(state["unsigned_activation"])
    else:
        from safetensors import safe_open

        from analyze_selected_weight_reconstruction import unpack_scale

        paths = list(FLUX_CACHE.glob("*/transformer_blocks.safetensors"))
        if len(paths) != 1:
            raise RuntimeError(f"expected one FLUX checkpoint, got {paths}")
        with safe_open(paths[0], framework="pt", device="cpu") as handle:
            for info in metadata.values():
                packed = handle.get_tensor(f"{info['quant_key']}.smooth")
                info["smooth"] = unpack_scale(
                    packed, packed.numel(), 1
                ).flatten().float()
                info["shift"] = 0.171875 if info["type"] == "ffn_down" else 0.0
                info["unsigned"] = info["type"] == "ffn_down"


def register_hooks(
    transformer,
    records: dict[str, list[torch.Tensor]],
    a4_records: dict[str, list[dict]],
    metadata: dict,
    calls_per_step: int,
):
    modules = dict(transformer.named_modules())
    handles = []
    call_counts = {key: 0 for key in metadata}

    for key, info in metadata.items():
        module = modules[info["name"]]

        def out_hook(_module, _inputs, output, *, key=key):
            tensor = output[0] if isinstance(output, tuple) else output
            channels = tensor.shape[-1]
            value = (
                tensor.detach()
                .float()
                .abs()
                .reshape(-1, channels)
                .amax(dim=0)
                .to("cpu")
            )
            records[key].append(value)

        def pre_hook(_module, inputs, *, key=key, info=info):
            call = call_counts[key]
            call_counts[key] += 1
            step = call // calls_per_step
            branch = call % calls_per_step
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(A4_ROWS, x.shape[0])
            index = torch.linspace(0, x.shape[0] - 1, count, device=x.device).long()
            sample = x.index_select(0, index).detach().float().cpu()
            if info["shift"]:
                sample = sample + info["shift"]
            metrics = quant_metrics(
                sample, info["smooth"].float().cpu(), info["unsigned"]
            )
            metrics.update(
                {
                    "key": key,
                    "type": info["type"],
                    "block": info["block"],
                    "step": step + 1,
                    "cfg_branch": branch if calls_per_step == 2 else None,
                    "sampled_tokens": count,
                }
            )
            a4_records[key].append(metrics)

        handles.append(module.register_forward_hook(out_hook))
        handles.append(module.register_forward_pre_hook(pre_hook))
    return handles


@torch.inference_mode()
def collect(model: str, output_dir: Path) -> None:
    metadata = names_for(model)
    load_smooth(model, metadata)
    records = {key: [] for key in metadata}
    a4_records = {key: [] for key in metadata}
    calls_per_step = 2 if model == "wan" else 1

    if model == "wan":
        from diffusers import UniPCMultistepScheduler, WanPipeline

        pipe = WanPipeline.from_pretrained(str(WAN_MODEL), torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=3.0
        )
        handles = register_hooks(
            pipe.transformer, records, a4_records, metadata, calls_per_step
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
        handles = register_hooks(
            pipe.transformer, records, a4_records, metadata, calls_per_step
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

    payload = {
        "model": model,
        "definition": "log2(max over batch,tokens[,CFG branches] of abs(layer output))",
        "steps": 50,
        "seed": 44,
        "calls_per_step": calls_per_step,
        "layers": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("smooth",)}
            for k, v in metadata.items()
        },
        "surfaces": surfaces,
    }
    path = output_dir / f"{model}_multilayer_surfaces.pt"
    torch.save(payload, path)

    # Aggregate A4: average CFG branches when present; keep per-step mse/nmse.
    a4_summary = []
    for key, rows in a4_records.items():
        by_step: dict[int, list[dict]] = {}
        for row in rows:
            by_step.setdefault(row["step"], []).append(row)
        for step, values in sorted(by_step.items()):
            a4_summary.append(
                {
                    "model": model,
                    "key": key,
                    "type": metadata[key]["type"],
                    "block": metadata[key]["block"],
                    "step": step,
                    "mse": float(sum(v["mse"] for v in values) / len(values)),
                    "nmse": float(sum(v["nmse"] for v in values) / len(values)),
                    "max_over_rms": float(
                        sum(v["max_over_rms"] for v in values) / len(values)
                    ),
                    "num_branches": len(values),
                }
            )
    a4_path = output_dir / f"{model}_a4_timestep_metrics.json"
    a4_path.write_text(json.dumps(a4_summary, indent=2), encoding="utf-8")

    summary = {
        k: {
            **{kk: vv for kk, vv in v.items() if kk not in ("smooth",)},
            "shape": list(surfaces[k].shape),
        }
        for k, v in metadata.items()
    }
    (output_dir / f"{model}_multilayer_metadata.json").write_text(
        json.dumps(
            {
                "model": model,
                "path": str(path),
                "a4_path": str(a4_path),
                "definition": payload["definition"],
                "layers": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"layers": list(summary), "a4_rows": len(a4_summary)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux"), required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/timestep_outliers_multilayer")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collect(args.model, args.output_dir)


if __name__ == "__main__":
    main()
