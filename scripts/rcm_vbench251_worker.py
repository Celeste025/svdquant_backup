#!/usr/bin/env python3
"""Generate one rCM-Wan variant for a resumable VBench-251 selection."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data1/models/svdquant-wjq")
BASE = DATA / "models/Wan2.1-T2V-1.3B-Diffusers"
RCM_TRANSFORMER = DATA / "models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer"
SVDQUANT_CKPT = DATA / "ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16"
INT4_SVDQUANT_CKPT = DATA / "ckpts/rcm-wan2.1-1.3b-int4-s16-g10"
DIMENSIONS = (
    "aesthetic_quality", "scene", "imaging_quality", "overall_consistency",
    "background_consistency", "subject_consistency", "dynamic_degree", "motion_smoothness",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_pure_w4a4(pipe: WanPipeline, quant_config: str) -> None:
    """Install a dynamic W4A4 baseline without calibration, smoothing, or low rank."""
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    diffusion_root = ROOT / "third_party/deepcompressor/examples/diffusion"
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(diffusion_root)
        sys.argv = [
            "rcm_vbench251_worker.py", "configs/model/wan2.1-1.3b.yaml",
            quant_config, "configs/svdquant/rcm_wan_pure_nvfp4.yaml",
            f"--pipeline-path={BASE}", "--skip-eval", "--skip-gen",
        ]
        config, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
        if config.quant.enabled_smooth or config.quant.wgts.enabled_low_rank:
            raise RuntimeError("pure W4A4 recipe unexpectedly enabled smoothing or low rank")
        ptq(DiffusionModelStruct.construct(pipe), config.quant, cache=None)
    finally:
        sys.argv, _ = old_argv, os.chdir(old_cwd)


def load_svdquant(pipe: WanPipeline, checkpoint: Path, recipe: str | None = None) -> None:
    # The existing inference helper reconstructs the checkpoint's real-NVFP4+
    # SVDQuant structure before loading its saved model and scale tensors.
    from infer_rcm_wan_4step import load_quantized_transformer
    load_quantized_transformer(pipe, checkpoint, BASE, recipe)


def validate_variant(pipe: WanPipeline, variant: str) -> dict[str, int]:
    # rCM has six condition-embedder Linears in addition to the 30 x 10
    # transformer-block projections targeted by the quantization recipe.
    linears = [(n, m) for n, m in pipe.transformer.named_modules() if isinstance(m, torch.nn.Linear)]
    targets = [(n, m) for n, m in linears if n.startswith("blocks.")]
    if len(targets) != 300:
        raise RuntimeError(f"expected 300 quantization-target Linears, found {len(targets)} (total={len(linears)})")
    lowrank = sum(
        1 for m in pipe.transformer.modules() if type(m).__name__ == "LowRankBranch"
    )
    if variant == "nvfp4" and lowrank:
        raise RuntimeError(f"pure NVFP4 unexpectedly has {lowrank} LowRankBranch modules")
    return {"linear_modules": len(linears), "quantization_target_linears": len(targets), "lowrank_modules": lowrank}


def decode_spatial_tiled(vae, latents: torch.Tensor, core: int = 20, halo: int = 16) -> torch.Tensor:
    _, _, _, latent_h, latent_w = latents.shape
    output = None
    for core_y0 in range(0, latent_h, core):
        core_y1 = min(core_y0 + core, latent_h)
        tile_y0, tile_y1 = max(0, core_y0 - halo), min(latent_h, core_y1 + halo)
        for core_x0 in range(0, latent_w, core):
            core_x1 = min(core_x0 + core, latent_w)
            tile_x0, tile_x1 = max(0, core_x0 - halo), min(latent_w, core_x1 + halo)
            tile = vae.decode(latents[:, :, :, tile_y0:tile_y1, tile_x0:tile_x1], return_dict=False)[0]
            scale_y, scale_x = tile.shape[-2] // (tile_y1 - tile_y0), tile.shape[-1] // (tile_x1 - tile_x0)
            if output is None:
                output = torch.empty((*tile.shape[:-2], latent_h * scale_y, latent_w * scale_x), dtype=torch.float32, device="cpu")
            output[:, :, :, core_y0 * scale_y:core_y1 * scale_y, core_x0 * scale_x:core_x1 * scale_x].copy_(
                tile[:, :, :, (core_y0-tile_y0)*scale_y:(core_y1-tile_y0)*scale_y, (core_x0-tile_x0)*scale_x:(core_x1-tile_x0)*scale_x].float().cpu()
            )
            del tile
            torch.cuda.empty_cache()
    assert output is not None
    return output


def complete(video: Path, status: Path) -> bool:
    if not video.is_file() or not status.is_file():
        return False
    try:
        return json.loads(status.read_text()).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def generate(pipe: WanPipeline, case: dict, video: Path) -> None:
    device = torch.device("cuda")
    # Keep this large encoder off GPU between cases.  It must be colocated with
    # input_ids during prompt encoding, then the DiT owns the GPU for sampling.
    pipe.text_encoder.to(device)
    prompt_embeds, _ = pipe.encode_prompt(prompt=case["prompt"], do_classifier_free_guidance=False,
                                          max_sequence_length=512, device=device, dtype=pipe.transformer.dtype)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(0)
    latents = pipe.prepare_latents(batch_size=1, num_channels_latents=pipe.transformer.config.in_channels,
        height=480, width=832, num_frames=77, dtype=torch.float32, device=device, generator=generator)
    import math
    trig_steps = torch.tensor([math.atan(80.0), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    t_steps = torch.sin(trig_steps) / (torch.cos(trig_steps) + torch.sin(trig_steps))
    latents = latents.to(torch.float64) * t_steps[0]
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        timestep = (t_cur.float() * ones * 1000).to(device=device, dtype=pipe.transformer.dtype)
        with torch.no_grad():
            velocity = pipe.transformer(hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                                        encoder_hidden_states=prompt_embeds, return_dict=False)[0].to(torch.float64)
        latents = (1 - t_next) * (latents - t_cur * velocity) + t_next * torch.randn(
            latents.shape, dtype=torch.float32, device=device, generator=generator)
    pipe.transformer.to("cpu")
    del velocity, prompt_embeds
    gc.collect(); torch.cuda.empty_cache()
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=pipe.vae.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=pipe.vae.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    with torch.inference_mode():
        decoded = decode_spatial_tiled(pipe.vae, latents.float().to(pipe.vae.dtype) / std + mean)
    frames = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
    video.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(video), fps=16)
    pipe.transformer.to(device)
    del decoded, frames, latents
    gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("bf16", "nvfp4", "nvfp4_svdquant", "int4_plain", "int4_svdquant"), required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    # The original 251-prompt manifest names its selected subset ``pilot_cases``;
    # the dedicated INT4 VBench-51 manifest calls the same list ``cases``.
    selected_cases = manifest.get("pilot_cases", manifest.get("cases"))
    if not selected_cases:
        raise RuntimeError("manifest contains no generation cases")
    cases = selected_cases[:1] if args.smoke else selected_cases
    pipe = WanPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    if args.variant == "nvfp4":
        load_pure_w4a4(pipe, "configs/svdquant/real_nvfp4.yaml")
    elif args.variant == "int4_plain":
        load_pure_w4a4(pipe, "configs/svdquant/int4.yaml")
    elif args.variant == "nvfp4_svdquant":
        load_svdquant(pipe, SVDQUANT_CKPT, "real-nvfp4")
    elif args.variant == "int4_svdquant":
        load_svdquant(pipe, INT4_SVDQUANT_CKPT, "rcm-wan-int4-svdquant-v1")
    validation = validate_variant(pipe, args.variant)
    pipe.text_encoder.to("cpu"); pipe.vae.to("cuda"); torch.cuda.empty_cache()
    worker_status = args.output / "worker_status" / f"{args.variant}.json"
    worker_status.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(worker_status, {"state": "running", "variant": args.variant, "validation": validation, "pid": os.getpid(), "started_at": time.time()})
    failures, generated, skipped = [], 0, 0
    for case in cases:
        case_dir = args.output / "cases" / case["case_id"]
        video, status = case_dir / f"{args.variant}.mp4", case_dir / f"{args.variant}.json"
        if complete(video, status):
            skipped += 1
            continue
        started = time.time()
        try:
            generate(pipe, case, video)
            atomic_json(status, {"state": "complete", "variant": args.variant, "case_id": case["case_id"],
                "prompt": case["prompt"], "seed": 0, "height": 480, "width": 832, "frames": 77,
                "steps": 4, "sigma_max": 80.0, "guidance": 0.0, "video": str(video), "seconds": time.time()-started})
            generated += 1
        except Exception as exc:
            failure = {"case_id": case["case_id"], "error": repr(exc)}
            failures.append(failure)
            atomic_json(status, {"state": "failed", "variant": args.variant, **failure})
            torch.cuda.empty_cache()
    atomic_json(worker_status, {"state": "complete" if not failures else "completed_with_failures", "variant": args.variant,
        "validation": validation, "generated": generated, "skipped": skipped, "failures": failures, "finished_at": time.time()})
    if failures:
        raise SystemExit(f"{args.variant}: {len(failures)} case(s) failed")


if __name__ == "__main__":
    main()
