#!/usr/bin/env python3
"""Smooth-aware BF16 restoration of the highest-NMSE rCM-Wan Linear layers.

Existing SmoothQuant coordinate transforms stay in place. Each restored Linear
receives the BF16 weight in that coordinate, while direct QDQ and low-rank
hooks are removed. Every replacement is locally verified before sampling.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import WanPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/models/svdquant-wjq")
MODEL = DATA / "models" / "rcm-Wan2.1-T2V-1.3B-Diffusers"
CKPT = DATA / "ckpts" / "rcm-wan2.1-1.3b-real-nvfp4-s16"
RECORDS = ROOT / "results/reports/linear_online_corrected_4x10/rcm-wan2.1-1.3b_online_records.json"
TRAJECTORY_RECORDS = ROOT / "results/reports/rcm_trajectory_sensitivity_audit/all300_2prompt_actual_normalized_1pct/trajectory_impact.csv"
SIGNED_RECORDS = ROOT / "results/reports/rcm_signed_alignment_audit/airplane_seed303/per_layer_signed_summary.csv"
PROMPT = "An airplane soaring through a clear blue sky above white clouds."


def load_helper():
    path = ROOT / "scripts/infer_rcm_wan_4step.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def select_layers(records: Path, fraction: float, selection: str = "nmse", random_seed: int = 0) -> list[str]:
    if not 0 < fraction <= 1:
        raise ValueError(f"top fraction must be in (0, 1], got {fraction}")
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in json.loads(records.read_text())["records"]:
        total = totals[row["layer"]]
        total[0] += float(row["sum_sq_err"])
        total[1] += float(row["sum_ref_sq"])
    if selection == "attn":
        return sorted(name for name in totals if ".attn1." in name or ".attn2." in name)
    if selection == "ffn":
        return sorted(name for name in totals if ".ffn." in name)
    count = max(1, math.ceil(fraction * len(totals)))
    if selection == "random":
        return sorted(random.Random(random_seed).sample(sorted(totals), count))
    ranked = sorted(((err / max(ref, 1e-30), name) for name, (err, ref) in totals.items()), reverse=True)
    return [name for _, name in ranked[:count]]


def select_trajectory_actual_layers(records: Path, fraction: float) -> tuple[list[str], dict[str, float]]:
    """Rank by equal-weight mean actual final-latent NMSE over prompt/timestep."""
    values: dict[str, list[float]] = defaultdict(list)
    with records.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "actual":
                values[row["layer"]].append(float(row["final_latent_nmse"]))
    if len(values) != 300 or any(len(items) != 8 for items in values.values()):
        counts = sorted({len(items) for items in values.values()})
        raise RuntimeError(f"expected 300 layers x 8 actual interventions, got {len(values)} layers with counts {counts}")
    scores = {name: sum(items) / len(items) for name, items in values.items()}
    count = max(1, math.ceil(fraction * len(scores)))
    ranked = sorted(scores, key=lambda name: (-scores[name], name))
    return ranked[:count], scores


def select_signed_layers(records: Path, fraction: float, bottom: bool = False) -> tuple[list[str], dict[str, float]]:
    """Rank smooth-aware restores by predicted signed final-NMSE benefit."""
    scores: dict[str, float] = {}
    with records.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scores[row["layer"]] = float(row["predicted_nmse_reduction_sum"])
    if len(scores) != 300:
        raise RuntimeError(f"expected signed scores for 300 layers, found {len(scores)}")
    count = max(1, math.ceil(fraction * len(scores)))
    ranked = sorted(scores, key=lambda name: (scores[name], name) if bottom else (-scores[name], name))
    return ranked[:count], scores


def _smooth_hooks(module: torch.nn.Module, in_features: int):
    matches = []
    for hook in module._forward_pre_hooks.values():
        processor = getattr(hook, "processor", None)
        scale = getattr(processor, "smooth_scale", None)
        if type(processor).__name__ == "ActivationSmoother" and torch.is_tensor(scale) and scale.numel() == in_features:
            matches.append(processor)
    return matches


def input_smoother(root: torch.nn.Module, name: str, linear: torch.nn.Linear):
    """Return (smoother, is_direct) for rCM's actual input coordinate."""
    direct = _smooth_hooks(linear, linear.in_features)
    if direct:
        if len(direct) != 1:
            raise RuntimeError(f"{name}: expected one direct smooth hook, found {len(direct)}")
        return direct[0], True
    if name.endswith((".to_q", ".to_k", ".to_v")) and ".attn" in name:
        attn_name, suffix = name.rsplit(".", 1)
        if attn_name.endswith(".attn1") or suffix == "to_q":
            external = _smooth_hooks(root.get_submodule(attn_name), linear.in_features)
            if len(external) == 1:
                return external[0], False
            if external:
                raise RuntimeError(f"{name}: expected one parent smooth hook, found {len(external)}")
    return None, False


def remove_direct_qdq_and_lowrank(module: torch.nn.Module, name: str) -> dict[str, int]:
    result = {"removed_pre_hooks": 0, "removed_post_hooks": 0}
    for attr, report_key in (("_forward_pre_hooks", "removed_pre_hooks"), ("_forward_hooks", "removed_post_hooks")):
        hooks = getattr(module, attr)
        for key, hook in list(hooks.items()):
            processor = getattr(hook, "processor", None)
            branch = getattr(hook, "branch", None)
            processor_name = type(processor).__name__ if processor is not None else ""
            branch_name = type(branch).__name__ if branch is not None else ""
            if "Quantizer" in processor_name or branch_name == "LowRankBranch":
                del hooks[key]
                result[report_key] += 1
    return result


@torch.inference_mode()
def restore_one(root: torch.nn.Module, name: str, state: dict[str, torch.Tensor]) -> dict[str, float | int | bool]:
    module = root.get_submodule(name)
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"{name}: expected nn.Linear, got {type(module)}")
    smoother, is_direct = input_smoother(root, name, module)
    if smoother is None:
        weight_scale = 1.0
    else:
        scale = smoother.smooth_scale.to(device=module.weight.device, dtype=module.weight.dtype).view(1, -1)
        # The standard hook applies x/s, so W_bf16*s preserves W_bf16*x.
        weight_scale = scale.reciprocal() if bool(getattr(smoother, "upscale", False)) else scale
    module.weight.copy_(state["weight"].to(device=module.weight.device, dtype=module.weight.dtype) * weight_scale)
    if module.bias is not None:
        if "bias" not in state:
            raise RuntimeError(f"{name}: BF16 source has no bias")
        module.bias.copy_(state["bias"].to(device=module.bias.device, dtype=module.bias.dtype))
    report = remove_direct_qdq_and_lowrank(module, name)

    # Fail closed unless the transformed module is locally equivalent to BF16.
    x = torch.randn((1, 3, module.in_features), device=module.weight.device, dtype=module.weight.dtype)
    got = module(x if smoother is None or is_direct else smoother.process(x)).float()
    bias = state["bias"].to(device=x.device, dtype=x.dtype) if "bias" in state else None
    ref = F.linear(x, state["weight"].to(device=x.device, dtype=x.dtype), bias).float()
    nmse = float((got - ref).square().sum() / ref.square().sum().clamp_min(1e-30))
    if nmse > 5e-5:
        raise RuntimeError(f"{name}: local BF16 equivalence failed (NMSE={nmse:.3g})")
    return report | {"smooth_owner": "direct" if is_direct else "parent" if smoother is not None else "none",
                     "smooth_upscale": bool(getattr(smoother, "upscale", False)), "local_nmse": nmse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-fraction", type=float, default=0.10, help="pooled-NMSE ranked fraction of Linear layers to restore")
    ap.add_argument(
        "--selection",
        choices=("nmse", "trajectory-actual", "signed-top", "signed-bottom", "random", "attn", "ffn"),
        default="nmse",
        help="nmse/random select a fraction; attn/ffn restore every recorded Linear in that component",
    )
    ap.add_argument("--random-seed", type=int, default=20260830)
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--trajectory-records", type=Path, default=TRAJECTORY_RECORDS)
    ap.add_argument("--signed-records", type=Path, default=SIGNED_RECORDS)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--output", type=Path, default=ROOT / "results/samples/rcm_restore_smoothaware/top10pct_airplane_seed303.mp4")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--latent-output", type=Path, help="optionally save the final pre-VAE latent")
    ap.add_argument("--skip-decode", action="store_true", help="stop after denoising; requires --latent-output")
    ap.add_argument("--verify-only", action="store_true", help="restore and write per-layer checks without sampling")
    args = ap.parse_args()
    if args.skip_decode and args.latent_output is None:
        ap.error("--skip-decode requires --latent-output")
    trajectory_scores = None
    signed_scores = None
    if args.selection == "trajectory-actual":
        keep, trajectory_scores = select_trajectory_actual_layers(args.trajectory_records, args.top_fraction)
    elif args.selection in ("signed-top", "signed-bottom"):
        keep, signed_scores = select_signed_layers(
            args.signed_records, args.top_fraction, bottom=args.selection == "signed-bottom"
        )
    else:
        keep = select_layers(args.records, args.top_fraction, args.selection, args.random_seed)
    selection_detail = f"{args.top_fraction:.1%}" if args.selection in (
        "nmse", "trajectory-actual", "signed-top", "signed-bottom", "random"
    ) else "all"
    print(f"Smooth-aware BF16 restore: {args.selection} {selection_detail} = {len(keep)} layers", flush=True)

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
        verify_path.write_text(json.dumps({"method": "smooth-aware in-place BF16 restore", "selection": args.selection, "random_seed": args.random_seed, "restored_layers": keep, "checks": checks}, indent=2) + "\n")
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

    if args.latent_output is not None:
        args.latent_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), args.latent_output)
    if args.skip_decode:
        latent_meta = {"method": "smooth-aware in-place BF16 restore", "top_fraction": args.top_fraction,
                       "selection": args.selection, "restored_layers": keep, "checks": checks,
                       "signed_records": str(args.signed_records) if signed_scores is not None else None,
                       "signed_scores": {name: signed_scores[name] for name in keep} if signed_scores is not None else None,
                       "prompt": args.prompt, "seed": args.seed, "latent_output": str(args.latent_output)}
        args.latent_output.with_suffix(".json").write_text(json.dumps(latent_meta, indent=2) + "\n")
        print(f"Saved {args.latent_output}", flush=True)
        return

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
    meta = {"method": "smooth-aware in-place BF16 restore", "top_fraction": args.top_fraction, "selection": args.selection, "random_seed": args.random_seed,
            "trajectory_records": str(args.trajectory_records) if args.selection == "trajectory-actual" else None,
            "trajectory_score_definition": "equal-weight mean actual final_latent_nmse over 2 prompts x 4 timesteps" if args.selection == "trajectory-actual" else None,
            "trajectory_scores": {name: trajectory_scores[name] for name in keep} if trajectory_scores is not None else None,
            "signed_records": str(args.signed_records) if signed_scores is not None else None,
            "signed_score_definition": "sum over 4 timesteps of -2<d(half final NMSE)/dy_q, y_bf16-y_q>" if signed_scores is not None else None,
            "signed_scores": {name: signed_scores[name] for name in keep} if signed_scores is not None else None,
            "restored_layers": keep, "checks": checks, "prompt": args.prompt, "seed": args.seed,
            "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
