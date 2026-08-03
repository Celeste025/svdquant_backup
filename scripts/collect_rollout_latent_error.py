#!/usr/bin/env python3
"""Collect model outputs and post-step latents for BF16/W4A4 trajectories.

Run each model in three modes:
  bf16:    reference trajectory
  quant:   natural quantized rollout
  forced:  quantized model, with the BF16 post-step latent restored after every step

The forced run therefore measures quantized model output error on the same latent
trajectory as BF16, while the natural quant run measures accumulated rollout error.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch


FLUX_BASE = "black-forest-labs/FLUX.1-dev"
FLUX_QUANT = "mit-han-lab/svdq-int4-flux.1-dev"
FLUX_PROMPT = "A cat holding a sign that says hello world"
WAN_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)
WAN_PROMPT = "An astronaut feeding ducks on a sunny afternoon, reflection from the water."
WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
WAN_QUANT = Path("outputs/wan_svdquant_calib_large/svdquant_large_calibrated.pt")
STEPS = 50
SEED = 44
# Human-readable steps 1, 6, ..., 46, 50 (zero-based indices below).
CAPTURE_INDICES = tuple(list(range(0, STEPS, 5)) + [STEPS - 1])


def seed_everything(seed: int) -> None:
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def install_scheduler_recorder(pipe, model_outputs: dict[int, torch.Tensor]):
    original = pipe.scheduler.step
    counter = {"i": 0}

    def wrapped(*args, **kwargs):
        index = counter["i"]
        model_output = args[0] if args else kwargs["model_output"]
        if index in CAPTURE_INDICES:
            model_outputs[index] = model_output.detach().to("cpu", torch.bfloat16).clone()
        result = original(*args, **kwargs)
        counter["i"] += 1
        return result

    pipe.scheduler.step = wrapped
    return counter


def make_callback(
    post_latents: dict[int, torch.Tensor],
    *,
    forced_reference: list[torch.Tensor] | None,
):
    def callback(_pipe, index, _timestep, callback_kwargs):
        latent = callback_kwargs["latents"]
        if index in CAPTURE_INDICES:
            post_latents[index] = latent.detach().to("cpu", torch.bfloat16).clone()
        if forced_reference is not None:
            # Restoring the BF16 post-step latent makes the next model invocation
            # teacher-forced. Step zero already starts from the identical noise.
            latent = forced_reference[index].to(device=latent.device, dtype=latent.dtype)
        return {"latents": latent}

    return callback


def load_pipe(model: str, mode: str):
    quantized = mode in {"quant", "forced"}
    if model == "wan":
        from compare_wan_svdquant_fake import (
            load_calibrated_transformer,
            make_pipeline as make_wan_pipeline,
        )

        pipe = make_wan_pipeline(WAN_MODEL)
        if quantized:
            pipe.transformer.to("cpu")
            converted, payload = load_calibrated_transformer(
                pipe.transformer, WAN_QUANT, quantize_activation=True
            )
            if len(converted) != 300:
                raise RuntimeError(f"expected 300 Wan quantized linears, got {len(converted)}")
            if (payload["group_size"], payload["rank"], payload.get("weight_bits", 4)) != (
                64,
                32,
                4,
            ):
                raise RuntimeError("Wan checkpoint is not W4A4 group64 rank32")
        pipe.enable_model_cpu_offload(gpu_id=0)
        return pipe

    from diffusers import FluxPipeline

    if quantized:
        from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            FLUX_QUANT, device="cuda:0"
        )
        return FluxPipeline.from_pretrained(
            FLUX_BASE, transformer=transformer, torch_dtype=torch.bfloat16
        ).to("cuda:0")
    pipe = FluxPipeline.from_pretrained(FLUX_BASE, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(gpu_id=0)
    return pipe


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan", "flux"), required=True)
    parser.add_argument("--mode", choices=("bf16", "quant", "forced"), required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rollout_latent_error")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = None
    if args.mode == "forced":
        bf16_path = args.output_dir / f"{args.model}_bf16.pt"
        payload = torch.load(bf16_path, map_location="cpu", weights_only=False)
        reference = payload["all_post_latents"]
        if len(reference) != STEPS:
            raise RuntimeError(f"invalid BF16 trajectory in {bf16_path}")

    pipe = load_pipe(args.model, args.mode)
    model_outputs: dict[int, torch.Tensor] = {}
    post_latents: dict[int, torch.Tensor] = {}
    all_post_latents: list[torch.Tensor] = []
    install_scheduler_recorder(pipe, model_outputs)

    def callback(pipe_, index, timestep, callback_kwargs):
        latent = callback_kwargs["latents"]
        if args.mode == "bf16":
            all_post_latents.append(
                latent.detach().to("cpu", torch.bfloat16).clone()
            )
        return make_callback(
            post_latents, forced_reference=reference
        )(pipe_, index, timestep, callback_kwargs)

    seed_everything(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    common = dict(
        num_inference_steps=STEPS,
        generator=generator,
        output_type="latent",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    if args.model == "wan":
        pipe(
            prompt=WAN_PROMPT,
            negative_prompt=WAN_NEGATIVE_PROMPT,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=6.0,
            **common,
        )
    else:
        pipe(
            FLUX_PROMPT,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            **common,
        )

    result = {
        "model": args.model,
        "mode": args.mode,
        "seed": SEED,
        "steps": STEPS,
        "capture_indices": CAPTURE_INDICES,
        "model_outputs": model_outputs,
        "post_latents": post_latents,
    }
    if args.mode == "bf16":
        result["all_post_latents"] = all_post_latents
    path = args.output_dir / f"{args.model}_{args.mode}.pt"
    torch.save(result, path)
    metadata = {
        "path": str(path),
        "model": args.model,
        "mode": args.mode,
        "seed": SEED,
        "steps": STEPS,
        "captured_human_steps": [i + 1 for i in CAPTURE_INDICES],
        "model_output_shapes": {str(k + 1): list(v.shape) for k, v in model_outputs.items()},
        "post_latent_shapes": {str(k + 1): list(v.shape) for k, v in post_latents.items()},
    }
    (args.output_dir / f"{args.model}_{args.mode}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
