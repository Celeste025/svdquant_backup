#!/usr/bin/env python3
"""Generate matched BF16 and fake SVDQuant Wan2.1 videos.

This is an accuracy prototype, not an efficient inference implementation.
Weights and activations are fake-quantized to signed INT4 with groups of 64.
A rank-32 BF16 branch is first extracted from the smoothed weight and only its
residual is quantized, matching DeepCompressor's default SVDQuant formulation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UniPCMultistepScheduler, WanPipeline
from diffusers.utils import export_to_video


DEFAULT_MODEL = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "pretrained_models/Wan2.1-T2V-1.3B-Diffusers"
)
DEFAULT_CALIB = Path(
    "/data/home/jinqiwen/workspace/video-distilation/DVDQuant_rep/"
    "results/viditq_w4a4/calib_data.pth"
)
DEFAULT_PROMPT = (
    "An astronaut feeding ducks on a sunny afternoon, reflection from the water."
)
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)
REMAIN_FP = ("condition_embedder", "patch_embedding", "proj_out", "scale_shift_table")


def fake_quant_s4_groupwise(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dynamic symmetric signed INT4 fake quantization along the last axis."""
    if x.shape[-1] % group_size:
        raise ValueError(f"last dimension {x.shape[-1]} is not divisible by {group_size}")
    shape = x.shape
    groups = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6).div_(7)
    groups = groups.div(scale).round_().clamp_(-7, 7).mul_(scale)
    return groups.reshape(shape).to(x.dtype)


def fake_quant_u4_groupwise(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dynamic unsigned INT4 fake quantization along the last axis."""
    if x.shape[-1] % group_size:
        raise ValueError(f"last dimension {x.shape[-1]} is not divisible by {group_size}")
    shape = x.shape
    groups = x.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
    scale = groups.amax(dim=-1, keepdim=True).clamp_min_(1e-6).div_(15)
    groups = groups.div(scale).round_().clamp_(0, 15).mul_(scale)
    return groups.reshape(shape).to(x.dtype)


class SVDQuantFakeLinear(nn.Module):
    """W4A4 fake linear plus a BF16 low-rank quantization-error branch."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        act_absmax: torch.Tensor,
        group_size: int,
        rank: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.rank = rank
        self.quantize_activation = True
        self.register_buffer("input_shift", torch.tensor(0, dtype=linear.weight.dtype))
        self.register_buffer("unsigned_activation", torch.tensor(False))

        weight = linear.weight.detach().to(device=device, dtype=torch.float32)
        x_span = act_absmax.float().amax(dim=0).to(device).clamp_min_(1e-6)
        w_span = weight.abs().amax(dim=0).clamp_min_(1e-6)
        # Standard SmoothQuant initialization. The transformed computation is
        # (x / smooth) @ (W * smooth)^T.
        smooth = (x_span / w_span).sqrt().clamp_(1e-4, 1e4)
        smooth_weight = weight * smooth.unsqueeze(0)

        effective_rank = min(rank, *smooth_weight.shape)
        # Randomized truncated SVD is used here to keep Wan's 300 projections
        # practical. It approximates the exact truncated SVD used by
        # DeepCompressor without changing the rank or compensated quantity.
        u, s, v = torch.svd_lowrank(smooth_weight, q=effective_rank, niter=4)
        up = u * s.unsqueeze(0)
        down = v.transpose(0, 1)
        low_rank_weight = up @ down
        qweight = fake_quant_s4_groupwise(smooth_weight - low_rank_weight, group_size)

        self.register_buffer("smooth", smooth.to(dtype=linear.weight.dtype, device="cpu"))
        self.register_buffer("qweight", qweight.to(dtype=linear.weight.dtype, device="cpu"))
        self.register_buffer("down", down.to(dtype=linear.weight.dtype, device="cpu"))
        self.register_buffer("up", up.to(dtype=linear.weight.dtype, device="cpu"))
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().to(device="cpu"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.input_shift
        x_smooth = x / self.smooth
        if self.quantize_activation:
            x_quant = (
                fake_quant_u4_groupwise(x_smooth, self.group_size)
                if bool(self.unsigned_activation)
                else fake_quant_s4_groupwise(x_smooth, self.group_size)
            )
        else:
            x_quant = x_smooth
        main = F.linear(x_quant, self.qweight, self.bias)
        # The low-rank branch consumes the unquantized, smoothed BF16 input.
        low_rank = F.linear(F.linear(x_smooth, self.down), self.up)
        return main + low_rank

    @classmethod
    def from_calibrated_state(
        cls,
        linear: nn.Linear,
        state: dict[str, torch.Tensor],
        *,
        group_size: int,
        rank: int,
        quantize_activation: bool = True,
    ) -> "SVDQuantFakeLinear":
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = linear.in_features
        module.out_features = linear.out_features
        module.group_size = group_size
        module.rank = rank
        module.quantize_activation = quantize_activation
        for key in ("smooth", "qweight", "down", "up"):
            module.register_buffer(key, state[key].to(device="cpu", dtype=linear.weight.dtype))
        input_shift = state.get("input_shift", torch.tensor(0, dtype=linear.weight.dtype))
        unsigned = state.get("unsigned_activation", torch.tensor(False))
        module.register_buffer("input_shift", input_shift.to(device="cpu", dtype=linear.weight.dtype))
        module.register_buffer("unsigned_activation", unsigned.to(device="cpu", dtype=torch.bool))
        if linear.bias is None:
            bias = torch.zeros(linear.out_features, dtype=torch.float64)
        else:
            bias = linear.bias.detach().to(device="cpu", dtype=torch.float64)
        if float(input_shift) != 0:
            correction = linear.weight.detach().to(device="cpu", dtype=torch.float64).sum(dim=1)
            bias = bias - correction * float(input_shift)
        module.register_buffer("bias", bias.to(dtype=linear.weight.dtype))
        return module


def set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


@torch.no_grad()
def convert_transformer(
    transformer: nn.Module,
    calib_path: Path,
    *,
    group_size: int,
    rank: int,
    device: torch.device,
) -> list[str]:
    calib = torch.load(calib_path, map_location="cpu", weights_only=False)
    targets = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, nn.Linear)
        and name in calib
        and not any(pattern in name for pattern in REMAIN_FP)
    ]
    converted: list[str] = []
    for index, (name, module) in enumerate(targets, 1):
        print(f"[quantize {index:03d}/{len(targets):03d}] {name}", flush=True)
        quantized = SVDQuantFakeLinear(
            module,
            act_absmax=calib[name],
            group_size=group_size,
            rank=rank,
            device=device,
        )
        set_submodule(transformer, name, quantized)
        converted.append(name)
        del module, quantized
        torch.cuda.empty_cache()
    if len(converted) != 300:
        raise RuntimeError(f"expected 300 quantized linears, got {len(converted)}")
    return converted


@torch.no_grad()
def load_calibrated_transformer(
    transformer: nn.Module,
    cache_path: Path,
    *,
    quantize_activation: bool = True,
    keep_bf16_patterns: tuple[str, ...] = (),
) -> tuple[list[str], dict]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "wan-svdquant-calibrated-v1":
        raise ValueError(f"unsupported calibration cache format in {cache_path}")
    state = payload["state"]
    converted = []
    for name in sorted(state):
        if any(pattern in name for pattern in keep_bf16_patterns):
            continue
        linear = transformer.get_submodule(name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name} is {type(linear)}, expected nn.Linear")
        quantized = SVDQuantFakeLinear.from_calibrated_state(
            linear,
            state[name],
            group_size=int(payload["group_size"]),
            rank=int(payload["rank"]),
            quantize_activation=quantize_activation,
        )
        set_submodule(transformer, name, quantized)
        converted.append(name)
    expected = sum(
        not any(pattern in name for pattern in keep_bf16_patterns) for name in state
    )
    if len(converted) != expected:
        raise RuntimeError(f"expected {expected} calibrated linears, got {len(converted)}")
    return converted, payload


def make_pipeline(model_path: Path) -> WanPipeline:
    pipe = WanPipeline.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    return pipe


def seed_everything(seed: int) -> None:
    """Match DVDQuant's global seeding helper exactly."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate(
    pipe: WanPipeline,
    output_path: Path,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    frames: int,
    steps: int,
    guidance: float,
    fps: int,
) -> float:
    pipe.enable_model_cpu_offload()
    seed_everything(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    started = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    elapsed = time.time() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(output_path), fps=fps)
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bf16", "quant"), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument(
        "--quant-cache",
        type=Path,
        default=Path("outputs/wan_svdquant_fake/svdquant_seed44_calibrated.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wan_svdquant_fake"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    # In DVDQuant's vbench_subset_8 run, astronaut is item index 2, hence
    # prompt_seed = base_seed(42) + index(2) = 44.
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--activation-mode", choices=("w4a4", "w4a16"), default="w4a4"
    )
    parser.add_argument(
        "--mixed-sensitive-bf16",
        action="store_true",
        help="Keep self-attention out, cross-attention Q, and FFN-down in BF16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_size != 64 or args.rank != 32:
        print(f"warning: requested group={args.group_size}, rank={args.rank}")
    pipe = make_pipeline(args.model)
    converted: list[str] = []
    calibration_summary = None
    cache_weight_bits = None
    cache_rank = None
    cache_group_size = None
    if args.mode == "quant":
        pipe.transformer.to("cpu")
        if args.quant_cache.exists():
            keep_bf16_patterns = (
                (".attn1.to_out.0", ".attn2.to_q", ".ffn.net.2")
                if args.mixed_sensitive_bf16
                else ()
            )
            converted, cache = load_calibrated_transformer(
                pipe.transformer,
                args.quant_cache,
                quantize_activation=args.activation_mode == "w4a4",
                keep_bf16_patterns=keep_bf16_patterns,
            )
            calibration_summary = cache.get("summary")
            cache_weight_bits = int(cache.get("weight_bits", 4))
            cache_rank = int(cache["rank"])
            cache_group_size = int(cache["group_size"])
        else:
            raise FileNotFoundError(
                f"calibrated cache not found: {args.quant_cache}; "
                "run collect_wan_svdquant_calib.py and calibrate_wan_svdquant.py first"
            )
    output_path = args.output_dir / f"{args.mode}_seed{args.seed}.mp4"
    elapsed = generate(
        pipe,
        output_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        frames=args.frames,
        steps=args.steps,
        guidance=args.guidance,
        fps=args.fps,
    )
    metadata = {
        "mode": args.mode,
        "model": str(args.model),
        "output": str(output_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "flow_shift": 3.0,
        "fps": args.fps,
        "weight_dtype": (
            f"sint{cache_weight_bits}_fake" if args.mode == "quant" else "bfloat16"
        ),
        "activation_dtype": (
            (
                "dynamic_int4_fake: signed except shifted FFN-down uses unsigned"
                if args.activation_mode == "w4a4"
                else "bfloat16 after SmoothQuant transform"
            )
            if args.mode == "quant"
            else "bfloat16"
        ),
        "group_size": cache_group_size if args.mode == "quant" else None,
        "svd_rank": cache_rank if args.mode == "quant" else None,
        "svd_target": (
            f"W_smooth; INT{cache_weight_bits} target is W_smooth - "
            f"SVD_rank{cache_rank}(W_smooth)"
            if args.mode == "quant"
            else None
        ),
        "quantized_linears": len(converted),
        "mixed_sensitive_bf16": args.mixed_sensitive_bf16,
        "calibration": calibration_summary,
        "elapsed_seconds": elapsed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.mode}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
