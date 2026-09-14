#!/usr/bin/env python3
"""Smooth-aware BF16 restoration for selected rCM-Wan attention projections.

This restores the *semantics* of one Q/K/V projection in a SmoothQuant graph:
the parent attention keeps x -> x/s, while the selected projection receives
W_bf16*s.  Direct QDQ and low-rank hooks on that projection are removed.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import WanPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/models/svdquant-wjq")
MODEL = DATA / "models" / "rcm-Wan2.1-T2V-1.3B-Diffusers"
CKPT = DATA / "ckpts" / "rcm-wan2.1-1.3b-real-nvfp4-s16"
RECORDS = ROOT / "results/reports/linear_online_4x10/rcm-wan2.1-1.3b_online_records.json"
PROMPT = "A cinematic shot of a red fox walking through a snowy forest at sunrise, soft fog drifting between the trees."


def load_helper():
    path = ROOT / "scripts/infer_rcm_wan_4step.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def select_layers(records: Path, threshold: float) -> list[str]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in json.loads(records.read_text())["records"]:
        values[row["layer"]].append(float(row["nmse"]))
    return sorted(name for name, xs in values.items() if sum(xs) / len(xs) > threshold)


def parent_smoother(root: torch.nn.Module, name: str, linear: torch.nn.Linear):
    if not (name.endswith((".to_q", ".to_k", ".to_v")) and ".attn" in name):
        raise ValueError(f"only attention to_q/to_k/to_v are supported, got {name}")
    parent = root.get_submodule(name.rsplit(".", 1)[0])
    matches = []
    for hook in parent._forward_pre_hooks.values():
        processor = getattr(hook, "processor", None)
        scale = getattr(processor, "smooth_scale", None)
        if type(processor).__name__ == "ActivationSmoother" and torch.is_tensor(scale) and scale.numel() == linear.in_features:
            matches.append(processor)
    if len(matches) != 1:
        raise RuntimeError(f"{name}: expected exactly one parent smooth hook, found {len(matches)}")
    return matches[0]


def remove_direct_qdq_and_lowrank(module: torch.nn.Module, name: str) -> dict[str, int]:
    result = {"removed_pre_hooks": 0, "removed_post_hooks": 0}
    for attr, report_key in (("_forward_pre_hooks", "removed_pre_hooks"), ("_forward_hooks", "removed_post_hooks")):
        hooks = getattr(module, attr)
        for key, hook in list(hooks.items()):
            processor = getattr(hook, "processor", None)
            branch = getattr(hook, "branch", None)
            processor_name = type(processor).__name__ if processor is not None else ""
            branch_name = type(branch).__name__ if branch is not None else ""
            if processor_name == "ActivationSmoother":
                raise RuntimeError(f"{name}: direct smooth hook requires graph-level handling")
            if "Quantizer" in processor_name or branch_name == "LowRankBranch":
                del hooks[key]
                result[report_key] += 1
    return result


@torch.inference_mode()
def restore_one(root: torch.nn.Module, name: str, state: dict[str, torch.Tensor]) -> dict[str, float | int | bool]:
    module = root.get_submodule(name)
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{name}: expected nn.Linear, got {type(module)}")
    smoother = parent_smoother(root, name, module)
    scale = smoother.smooth_scale.to(device=module.weight.device, dtype=module.weight.dtype).view(1, -1)
    # SmoothQuant's ordinary attention hook computes x/s.  If a future hook
    # uses x*s instead, use W/s to preserve W*x.
    weight_scale = scale.reciprocal() if bool(getattr(smoother, "upscale", False)) else scale
    module.weight.copy_(state["weight"].to(device=module.weight.device, dtype=module.weight.dtype) * weight_scale)
    if module.bias is not None:
        if "bias" not in state:
            raise RuntimeError(f"{name}: BF16 source has no bias")
        module.bias.copy_(state["bias"].to(device=module.bias.device, dtype=module.bias.dtype))
    report = remove_direct_qdq_and_lowrank(module, name)

    # Fail closed unless the transformed module is locally equivalent to BF16.
    x = torch.randn((1, 3, module.in_features), device=module.weight.device, dtype=module.weight.dtype)
    got = module(smoother.process(x)).float()
    bias = state["bias"].to(device=x.device, dtype=x.dtype) if "bias" in state else None
    ref = F.linear(x, state["weight"].to(device=x.device, dtype=x.dtype), bias).float()
    nmse = float((got - ref).square().sum() / ref.square().sum().clamp_min(1e-30))
    if nmse > 5e-5:
        raise RuntimeError(f"{name}: local BF16 equivalence failed (NMSE={nmse:.3g})")
    return report | {"smooth_upscale": bool(getattr(smoother, "upscale", False)), "local_nmse": nmse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=500.0, help="strict mean-NMSE threshold")
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=ROOT / "results/samples/rcm_restore_smoothaware/top1_seed42.mp4")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--verify-only", action="store_true", help="restore and write per-layer checks without sampling")
    args = ap.parse_args()
    keep = select_layers(args.records, args.threshold)
    print(f"Smooth-aware BF16 restore: {len(keep)} layers (NMSE > {args.threshold:g})", flush=True)

    # Retain only source tensors; the BF16 transformer itself is not present
    # during quantized inference.
    bf = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    bf_modules = dict(bf.transformer.named_modules())
    states = {}
    for name in keep:
        source = bf_modules.get(name)
        if not isinstance(source, torch.nn.Linear):
            raise RuntimeError(f"{name}: not a BF16 Linear")
        states[name] = {"weight": source.weight.detach().cpu().clone()}
        if source.bias is not None:
            states[name]["bias"] = source.bias.detach().cpu().clone()
    del bf, bf_modules
    gc.collect(); torch.cuda.empty_cache()

    helper = load_helper()
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    helper.load_quantized_transformer(pipe, CKPT, MODEL)
    checks = {}
    for name, state in states.items():
        try:
            checks[name] = restore_one(pipe.transformer, name, state)
        except Exception as exc:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            error_path = args.output.with_suffix(".verify.error.json")
            error_path.write_text(json.dumps({"failed_layer": name, "error": repr(exc), "completed_checks": checks}, indent=2) + "\n")
            raise
    if args.verify_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        verify_path = args.output.with_suffix(".verify.json")
        verify_path.write_text(json.dumps({"method": "smooth-aware in-place BF16 restore", "restored_layers": keep, "checks": checks}, indent=2) + "\n")
        print(f"Verified {len(keep)} layers -> {verify_path}", flush=True)
        return

    device = torch.device("cuda")
    embeds, _ = pipe.encode_prompt(prompt=args.prompt, do_classifier_free_guidance=False, max_sequence_length=512,
                                   device=device, dtype=pipe.transformer.dtype)
    pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, args.height, args.width, args.frames,
                                   torch.float32, device, generator)
    trig = torch.tensor([math.atan(80.0), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    times = torch.sin(trig) / (torch.cos(trig) + torch.sin(trig))
    latents = latents.to(torch.float64) * times[0]
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    for current, nxt in zip(times[:-1], times[1:]):
        timestep = (current.float() * ones * 1000).to(dtype=pipe.transformer.dtype)
        with torch.inference_mode():
            velocity = pipe.transformer(hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                                        encoder_hidden_states=embeds, return_dict=False)[0].to(torch.float64)
        latents = (1 - nxt) * (latents - current * velocity) + nxt * torch.randn(
            latents.shape, dtype=torch.float32, device=device, generator=generator)

    pipe.transformer.to("cpu"); del velocity; gc.collect(); torch.cuda.empty_cache()
    vae = pipe.vae
    latents = latents.float().to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    with torch.inference_mode():
        video = helper.decode_spatial_tiled(vae, latents / std + mean, core=100, halo=0)
    from diffusers.utils import export_to_video
    frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=16)
    meta = {"method": "smooth-aware in-place BF16 restore", "threshold_nmse": args.threshold,
            "restored_layers": keep, "checks": checks, "prompt": args.prompt, "seed": args.seed,
            "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
