#!/usr/bin/env python3
"""Run rCM's distilled 1--4 step schedule with a local Diffusers Wan DiT."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel

from exp_rcm_blockwise_lowrank_pilot import branch_map, freeze_all, install_activation_ste, restore_activation_processors
from exp_rcm_joint_smooth_lowrank import CKPT, MODEL, RCM_TRANSFORMER, DynamicSmooth
from diffusers.utils import export_to_video

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

def load_joint_state(student, dynamic, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    with torch.inference_mode():
        for index, branches in state["lowrank"].items():
            mappings = branch_map(student.blocks[int(index)])
            for name, values in branches.items():
                mappings[name].load_state_dict(values)
        if len(state["smooth_log_delta"]) != len(dynamic.delta):
            raise RuntimeError("smoothing group mismatch")
        for target, source in zip(dynamic.delta, state["smooth_log_delta"], strict=True):
            target.copy_(source.to(device=target.device, dtype=target.dtype))

def load_quantized_transformer(pipe, ckpt_dir: Path, model_path: Path) -> None:
    """Insert DeepCompressor quantizers and load an rCM-Wan PTQ checkpoint."""
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    diffusion_root = REPO_ROOT / "third_party" / "deepcompressor" / "examples" / "diffusion"
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(diffusion_root)
        sys.argv = [
            "infer_rcm_wan_4step.py",
            "configs/model/wan2.1-1.3b.yaml",
            "configs/svdquant/real_nvfp4.yaml",
            "configs/svdquant/wan_s16.yaml",
            f"--pipeline-path={model_path}",
            "--output-root=" + str(RESULTS_ROOT / "ptq_load_scratch"),
            "--cache-root=" + str(RESULTS_ROOT / "ptq_load_scratch"),
            "--skip-eval", "--skip-gen", "--eval-num-gpus=1",
        ]
        config, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
        model = DiffusionModelStruct.construct(pipe)
        ptq(model, config.quant, cache=None, load_dirpath=str(ckpt_dir), save_dirpath="", copy_on_save=False, save_model=False)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    gc.collect()
    torch.cuda.empty_cache()

def decode_spatial_tiled(vae, latents: torch.Tensor, core: int, halo: int) -> torch.Tensor:
    """Decode spatial tiles while preserving each tiles complete temporal cache."""
    _, _, _, latent_h, latent_w = latents.shape
    output = None
    for core_y0 in range(0, latent_h, core):
        core_y1 = min(core_y0 + core, latent_h)
        tile_y0, tile_y1 = max(0, core_y0 - halo), min(latent_h, core_y1 + halo)
        for core_x0 in range(0, latent_w, core):
            core_x1 = min(core_x0 + core, latent_w)
            tile_x0, tile_x1 = max(0, core_x0 - halo), min(latent_w, core_x1 + halo)
            tile = vae.decode(latents[:, :, :, tile_y0:tile_y1, tile_x0:tile_x1], return_dict=False)[0]
            scale_y = tile.shape[-2] // (tile_y1 - tile_y0)
            scale_x = tile.shape[-1] // (tile_x1 - tile_x0)
            if output is None:
                output = torch.empty((*tile.shape[:-2], latent_h * scale_y, latent_w * scale_x), dtype=torch.float32, device="cpu")
            crop_y0, crop_y1 = (core_y0 - tile_y0) * scale_y, (core_y1 - tile_y0) * scale_y
            crop_x0, crop_x1 = (core_x0 - tile_x0) * scale_x, (core_x1 - tile_x0) * scale_x
            output[:, :, :, core_y0 * scale_y:core_y1 * scale_y, core_x0 * scale_x:core_x1 * scale_x].copy_(tile[:, :, :, crop_y0:crop_y1, crop_x0:crop_x1].float().cpu())
            del tile
            torch.cuda.empty_cache()
    assert output is not None
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--quant-ckpt", type=Path, default=CKPT)
    parser.add_argument("--joint-state", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "samples" / "rcm-wan_bf16_vs_nvfp4" / "bf16_seed42.mp4")
    parser.add_argument("--steps", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--vae-core-latent-size", type=int, default=20)
    parser.add_argument("--vae-halo-latent-size", type=int, default=16)
    parser.add_argument("--latent-output", type=Path, help="optionally save the final pre-VAE latent")
    parser.add_argument("--skip-decode", action="store_true", help="stop after denoising; requires --latent-output")
    args = parser.parse_args()
    if args.skip_decode and args.latent_output is None:
        parser.error("--skip-decode requires --latent-output")

    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    pipe.to(device)
    load_quantized_transformer(pipe, args.quant_ckpt, MODEL)
    student = pipe.transformer; freeze_all(student)
    teacher = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16).to(device)
    freeze_all(teacher)
    dynamic = DynamicSmooth(student, teacher, args.quant_ckpt, bound=1.25).to(device)
    dynamic.teacher = None; teacher.to("cpu"); del teacher; gc.collect(); torch.cuda.empty_cache()
    originals, _ = install_activation_ste(student)
    load_joint_state(student, dynamic, args.joint_state)
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt,
        do_classifier_free_guidance=False,
        max_sequence_length=512,
        device=device,
        dtype=pipe.transformer.dtype,
    )
    # Keep only the active stage on GPU; VAE decode has a large 480p peak.
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.steps - 1]
    trig_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0.0], dtype=torch.float64, device=device)
    t_steps = torch.sin(trig_steps) / (torch.cos(trig_steps) + torch.sin(trig_steps))
    latents = latents.to(torch.float64) * t_steps[0]
    ones = torch.ones((latents.shape[0],), device=device, dtype=torch.float64)
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        timestep = (t_cur.float() * ones * 1000).to(device=device, dtype=pipe.transformer.dtype)
        with torch.no_grad():
            velocity = pipe.transformer(
                hidden_states=latents.to(pipe.transformer.dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0].to(torch.float64)
        latents = (1 - t_next) * (latents - t_cur * velocity) + t_next * torch.randn(
            latents.shape, dtype=torch.float32, device=device, generator=generator
        )

    if args.latent_output is not None:
        args.latent_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), args.latent_output)
    if args.skip_decode:
        report = vars(args) | {"schedule": "rCM TrigFlow->RectifiedFlow", "guidance": 0.0}
        args.latent_output.with_suffix(".json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in report.items()}, indent=2) + "\n")
        print(f"Saved {args.latent_output}")
        return

    pipe.transformer.to("cpu")
    del velocity
    gc.collect()
    torch.cuda.empty_cache()
    vae = pipe.vae
    latents = latents.float().to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    # VAE decode must be inference-only; otherwise its 81-frame autograd graph is retained.
    with torch.inference_mode():
        video = decode_spatial_tiled(
            vae, latents / std + mean, core=args.vae_core_latent_size, halo=args.vae_halo_latent_size
        )
    frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=16)
    report = vars(args) | {"schedule": "rCM TrigFlow->RectifiedFlow", "guidance": 0.0}
    args.output.with_suffix(".json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in report.items()}, indent=2) + "\n")
    restore_activation_processors(originals)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
