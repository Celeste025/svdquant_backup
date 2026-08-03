#!/usr/bin/env python3
"""Densely measure Wan first-step block 18--24 and output-head errors."""

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
BLOCKS = tuple(range(18, 25)) + (27, 29)


class StopAfterFirstStep(Exception):
    pass


def metrics(reference: torch.Tensor, quantized: torch.Tensor) -> dict:
    reference = reference.float()
    quantized = quantized.float()
    error = quantized - reference
    power = reference.square().mean().clamp_min(1e-20)
    return {
        "mse": float(error.square().mean()),
        "nmse": float(error.square().mean() / power),
        "reference_rms": float(power.sqrt()),
        "error_rms": float(error.square().mean().sqrt()),
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

    reference_path = ROOT / "wan_firststep_dense_head_bf16.pt"
    reference = None
    if args.mode == "quant":
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)[
            "outputs"
        ]

    calls: dict[str, int] = {}
    outputs: dict[str, torch.Tensor] = {}
    reports: list[dict] = []
    handles = []

    def record(stage: str, tensor: torch.Tensor) -> None:
        branch = calls.get(stage, 0)
        calls[stage] = branch + 1
        if branch >= 2:
            return
        key = f"{stage}_branch{branch}"
        cpu = tensor.detach().to("cpu", torch.bfloat16)
        if args.mode == "bf16":
            outputs[key] = cpu.clone()
        else:
            outputs[key] = cpu.clone()
            report = metrics(reference[key], cpu)
            report.update({"stage": stage, "cfg_branch": branch})
            reports.append(report)

    for block in BLOCKS:
        module = pipe.transformer.blocks[block]

        def block_hook(_module, _inputs, output, *, block=block):
            record(
                f"block{block}",
                output[0] if isinstance(output, tuple) else output,
            )

        handles.append(module.register_forward_hook(block_hook))

    handles.append(
        pipe.transformer.norm_out.register_forward_hook(
            lambda _module, _inputs, output: record("norm_out", output)
        )
    )
    handles.append(
        pipe.transformer.proj_out.register_forward_pre_hook(
            lambda _module, inputs: record("post_timestep_modulation", inputs[0])
        )
    )
    handles.append(
        pipe.transformer.proj_out.register_forward_hook(
            lambda _module, _inputs, output: record("proj_out", output)
        )
    )

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
    finally:
        for handle in handles:
            handle.remove()

    if args.mode == "bf16":
        torch.save(
            {"blocks": BLOCKS, "outputs": outputs},
            reference_path,
        )
        print(f"saved {len(outputs)} reference tensors to {reference_path}")
    else:
        # Wan CFG combines the unconditional and conditional model outputs as
        # uncond + guidance * (cond - uncond).  Small, correlated branch errors
        # can therefore be strongly amplified at guidance=6.
        for stage in ("proj_out",):
            ref_guided = reference[f"{stage}_branch0"].float() + 6.0 * (
                reference[f"{stage}_branch1"].float()
                - reference[f"{stage}_branch0"].float()
            )
            quant_guided = outputs[f"{stage}_branch0"].float() + 6.0 * (
                outputs[f"{stage}_branch1"].float()
                - outputs[f"{stage}_branch0"].float()
            )
            report = metrics(ref_guided, quant_guided)
            report.update({"stage": f"{stage}_cfg6_guided", "cfg_branch": None})
            reports.append(report)
        output_path = ROOT / "wan_firststep_dense_head_metrics.json"
        output_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps(reports, indent=2))
        print(f"saved metrics to {output_path}")


if __name__ == "__main__":
    main()
