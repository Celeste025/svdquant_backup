#!/usr/bin/env python3
"""Measure first-step cumulative residual-stream error through Wan blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from collect_timestep_activation_outliers import (
    WAN_MODEL,
    WAN_NEGATIVE,
    WAN_PROMPT,
    seed_everything,
)
from compare_wan_svdquant_fake import load_calibrated_transformer, make_pipeline


ROOT = Path("outputs/quant_error_diagnosis/block_propagation")
CACHE = Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt")
BLOCKS = tuple(list(range(0, 30, 3)) + [29])


class StopAfterFirstStep(Exception):
    pass


def metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    x, y = x.float(), y.float()
    e = y - x
    p = x.square().mean().clamp_min(1e-20)
    return {
        "mse": float(e.square().mean()),
        "nmse": float(e.square().mean() / p),
        "cosine": float(
            torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        ),
        "reference_rms": float(p.sqrt()),
        "error_rms": float(e.square().mean().sqrt()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16", "quant"), required=True)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    pipe = make_pipeline(WAN_MODEL)
    if args.mode == "quant":
        pipe.transformer.to("cpu")
        load_calibrated_transformer(pipe.transformer, CACHE, quantize_activation=True)
    modules = dict(pipe.transformer.named_modules())
    calls = {b: 0 for b in BLOCKS}
    reference = None
    if args.mode == "quant":
        reference = torch.load(
            ROOT / "wan_firststep_bf16.pt", map_location="cpu", weights_only=False
        )["outputs"]
    outputs = {}
    reports = []
    handles = []
    for block in BLOCKS:
        def hook(_module, _inputs, output, *, block=block):
            branch = calls[block]
            calls[block] += 1
            if branch >= 2:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            key = f"block{block}_branch{branch}"
            cpu = tensor.detach().to("cpu", torch.bfloat16)
            if args.mode == "bf16":
                outputs[key] = cpu.clone()
            else:
                report = metrics(reference[key], cpu)
                report.update({"block": block, "cfg_branch": branch})
                reports.append(report)

        handles.append(modules[f"blocks.{block}"].register_forward_hook(hook))

    pipe.enable_model_cpu_offload(gpu_id=0)
    seed_everything(44)

    def callback(_pipe, index, _timestep, kwargs):
        if index == 0:
            raise StopAfterFirstStep
        return kwargs

    try:
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
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    except StopAfterFirstStep:
        pass
    for handle in handles:
        handle.remove()
    if args.mode == "bf16":
        torch.save({"blocks": BLOCKS, "outputs": outputs}, ROOT / "wan_firststep_bf16.pt")
        print(f"saved {len(outputs)} reference tensors")
    else:
        (ROOT / "wan_firststep_metrics.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
