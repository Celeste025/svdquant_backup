#!/usr/bin/env python3
"""Finite-step validation of the signed Top-1 and Bottom-1 corrections."""
from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import WanPipeline

from exp_rcm_restore_top_fraction_bf16 import CKPT, MODEL, PROMPT, SIGNED_RECORDS, input_smoother, select_signed_layers
from infer_rcm_wan_4step import load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/rcm_signed_restore_top_bottom/step_size_check.json"
BF_LATENT = ROOT / "results/reports/rcm_trajectory_actual_restore_top10/latents/bf16.pt"


def run(pipe: WanPipeline, embeds: torch.Tensor, seed: int = 303) -> torch.Tensor:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, 480, 832, 81, torch.float32, device, generator)
    trig = torch.tensor([math.atan(80.0), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    times = torch.sin(trig) / (torch.cos(trig) + torch.sin(trig))
    latents = latents.to(torch.float64) * times[0]
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    for current, nxt in zip(times[:-1], times[1:]):
        timestep = (current.float() * ones * 1000).to(dtype=pipe.transformer.dtype)
        with torch.inference_mode():
            velocity = pipe.transformer(
                hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                encoder_hidden_states=embeds, return_dict=False,
            )[0].to(torch.float64)
        latents = (1 - nxt) * (latents - current * velocity) + nxt * torch.randn(
            latents.shape, dtype=torch.float32, device=device, generator=generator
        )
    return latents.detach().cpu()


def main() -> None:
    top, scores = select_signed_layers(SIGNED_RECORDS, 1 / 300)
    bottom, _ = select_signed_layers(SIGNED_RECORDS, 1 / 300, bottom=True)
    names = {"top1": top[0], "bottom1": bottom[0]}

    bf = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    bfmods = dict(bf.transformer.named_modules())
    states = {
        label: {
            "weight": bfmods[name].weight.detach().cpu().clone(),
            **({"bias": bfmods[name].bias.detach().cpu().clone()} if bfmods[name].bias is not None else {}),
        }
        for label, name in names.items()
    }
    del bf, bfmods
    gc.collect()

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    load_quantized_transformer(pipe, CKPT, MODEL)
    embeds, _ = pipe.encode_prompt(
        prompt=PROMPT, do_classifier_free_guidance=False, max_sequence_length=512,
        device="cuda", dtype=pipe.transformer.dtype,
    )
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    reference = torch.load(BF_LATENT, map_location="cpu", weights_only=True).float()
    ref2 = float(reference.square().sum())
    results: list[dict[str, Any]] = []

    baseline = run(pipe, embeds)
    baseline_nmse = float((baseline.float() - reference).square().sum()) / ref2
    results.append({"selection": "baseline", "layer": "", "alpha": 0.0, "latent_nmse": baseline_nmse,
                    "change_pct_vs_nvfp4": 0.0, "predicted_linear_nmse": baseline_nmse})

    for label, name in names.items():
        module = pipe.transformer.get_submodule(name)
        state = {key: value.to(device="cuda", dtype=torch.bfloat16) for key, value in states[label].items()}
        smoother, direct = input_smoother(pipe.transformer, name, module)
        raw_holder: dict[str, torch.Tensor] = {}
        alpha_holder = {"value": 0.0}

        def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            raw_holder["input"] = args[0]

        def forward_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            raw = raw_holder.pop("input")
            weight = state["weight"]
            if smoother is not None and not direct:
                scale = smoother.smooth_scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)
                weight = weight * (scale.reciprocal() if bool(getattr(smoother, "upscale", False)) else scale)
            restored = F.linear(raw, weight, state.get("bias"))
            return (output.float() + alpha_holder["value"] * (restored.float() - output.float())).to(output.dtype)

        pre_handle = module.register_forward_pre_hook(pre_hook, prepend=True)
        post_handle = module.register_forward_hook(forward_hook)
        for alpha in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0):
            alpha_holder["value"] = alpha
            latent = run(pipe, embeds)
            nmse = float((latent.float() - reference).square().sum()) / ref2
            predicted = baseline_nmse - alpha * scores[name]
            results.append({
                "selection": label,
                "layer": name,
                "alpha": alpha,
                "signed_score_full_step": scores[name],
                "latent_nmse": nmse,
                "change_pct_vs_nvfp4": 100 * (nmse - baseline_nmse) / baseline_nmse,
                "predicted_linear_nmse": predicted,
            })
            print(label, alpha, nmse, flush=True)
        pre_handle.remove()
        post_handle.remove()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    with OUT.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in results for key in row}))
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
