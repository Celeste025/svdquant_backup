#!/usr/bin/env python3
"""Paired BF16 Q/K intervention experiment for rCM-Wan W4A4/NVFP4.

This is an oracle-style diagnostic, not a new quantization method.  It restores
selected Q/K Linear projections in their existing SmoothQuant coordinate while
leaving all non-selected quantized layers untouched.  BF16 reference tensors
are retained in CPU memory only for online per-step comparison; no activations,
denoiser outputs, Q/K tensors, or logits are written to disk.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

from eval_rcm_restore_video_similarity import compare as compare_video
from exp_rcm_restore_top_fraction_bf16 import CKPT, MODEL, PROMPT, load_helper, restore_one


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "samples" / "rcm_qk_bf16_oracle" / "airplane_seed303"
CASE_ORDER = ("nvfp4", "self-qk", "cross-qk", "all-qk")


def qk_case_layers(transformer: torch.nn.Module) -> dict[str, list[str]]:
    """Return exact self/cross QK groups from the loaded Wan transformer."""
    names = [name for name, module in transformer.named_modules() if isinstance(module, torch.nn.Linear)]
    self_qk = sorted(name for name in names if ".attn1." in name and name.endswith((".to_q", ".to_k")))
    cross_qk = sorted(name for name in names if ".attn2." in name and name.endswith((".to_q", ".to_k")))
    all_qk = sorted(set(self_qk) | set(cross_qk))
    if len(self_qk) != 60 or len(cross_qk) != 60 or len(all_qk) != 120:
        raise RuntimeError(
            "Unexpected Q/K topology: "
            f"self={len(self_qk)} (expected 60), cross={len(cross_qk)} (expected 60), "
            f"all={len(all_qk)} (expected 120)"
        )
    if set(self_qk) & set(cross_qk) or set(all_qk) != set(self_qk) | set(cross_qk):
        raise RuntimeError("Q/K case groups are not disjoint exact unions")
    return {"self-qk": self_qk, "cross-qk": cross_qk, "all-qk": all_qk}


def source_states(transformer: torch.nn.Module, names: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """Copy only selected BF16 Q/K source weights to CPU."""
    modules = dict(transformer.named_modules())
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name in names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"{name}: expected BF16 nn.Linear, got {type(module)}")
        state = {"weight": module.weight.detach().cpu().clone()}
        if module.bias is not None:
            state["bias"] = module.bias.detach().cpu().clone()
        result[name] = state
    return result


def _hook_signature(module: torch.nn.Module) -> tuple[tuple[str, str, str], ...]:
    records = []
    for hook in module._forward_pre_hooks.values():
        records.append(("pre", type(getattr(hook, "processor", None)).__name__, type(getattr(hook, "branch", None)).__name__))
    for hook in module._forward_hooks.values():
        records.append(("post", type(getattr(hook, "processor", None)).__name__, type(getattr(hook, "branch", None)).__name__))
    return tuple(records)


@torch.inference_mode()
def untouched_fingerprint(transformer: torch.nn.Module) -> dict[str, tuple[float, float, float, tuple[tuple[str, str, str], ...]]]:
    """Compactly fingerprint V/O/FFN modules that a Q/K restore must not alter."""
    result = {}
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        is_v = name.endswith(".to_v")
        is_o = ".attn" in name and (".to_out" in name or name.endswith(".out_proj"))
        is_ffn = ".ffn." in name or ".ff." in name or ".mlp." in name
        if not (is_v or is_o or is_ffn):
            continue
        weight = module.weight.float()
        bias_sum = 0.0 if module.bias is None else float(module.bias.float().sum())
        result[name] = (float(weight.sum()), float(weight.square().sum()), bias_sum, _hook_signature(module))
    if not result:
        raise RuntimeError("No V/O/FFN Linear modules found for integrity fingerprint")
    return result


def validate_untouched(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    if before.keys() != after.keys():
        raise RuntimeError("V/O/FFN module set changed during Q/K restoration")
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise RuntimeError(f"Q/K restoration modified non-target V/O/FFN modules: {changed[:5]}")
    return {"modules_checked": len(before), "modules_changed": 0}


def tensor_metrics(got: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Return pooled scalar metrics without retaining either tensor."""
    got = got.float()
    reference = reference.to(device=got.device, dtype=torch.float32)
    error = got - reference
    sum_sq_err = float(error.square().sum())
    sum_ref_sq = float(reference.square().sum())
    numel = int(reference.numel())
    cosine = float((got * reference).sum() / (got.square().sum().sqrt() * reference.square().sum().sqrt()).clamp_min(1e-30))
    return {
        "numel": numel,
        "mse": sum_sq_err / max(numel, 1),
        "nmse": sum_sq_err / max(sum_ref_sq, 1e-30),
        "cosine": cosine,
    }


def rcm_times(device: torch.device) -> torch.Tensor:
    trig = torch.tensor([math.atan(80.0), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    return torch.sin(trig) / (torch.cos(trig) + torch.sin(trig))


@torch.inference_mode()
def decode_and_save(pipe: WanPipeline, helper: Any, latents: torch.Tensor, output: Path) -> None:
    vae = pipe.vae
    device = latents.device
    latents = latents.float().to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    video = helper.decode_spatial_tiled(vae, latents / std + mean, core=100, halo=0)
    frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output), fps=16)


@torch.inference_mode()
def build_bf16_reference(
    prompt: str, seed: int, height: int, width: int, frames: int, output: Path, helper: Any
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]], dict[str, list[str]]]:
    """Run BF16 once and keep only needed paired tensors in CPU RAM."""
    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    groups = qk_case_layers(pipe.transformer)
    states = source_states(pipe.transformer, groups["all-qk"])
    embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False, max_sequence_length=512,
                                   device=device, dtype=pipe.transformer.dtype)
    embeds_cpu = embeds.detach().cpu()
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels, height, width, frames,
                                   torch.float32, device, generator).to(torch.float64)
    times = rcm_times(device)
    latents.mul_(times[0])
    noises = [torch.randn(latents.shape, dtype=torch.float32, device=device, generator=generator).cpu() for _ in range(4)]
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    reference_inputs, reference_velocities, reference_latents = [], [], [latents.detach().cpu()]
    for step, (current, nxt) in enumerate(zip(times[:-1], times[1:], strict=True)):
        reference_inputs.append(latents.detach().cpu())
        timestep = (current.float() * ones * 1000).to(dtype=pipe.transformer.dtype)
        velocity = pipe.transformer(hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                                    encoder_hidden_states=embeds, return_dict=False)[0].to(torch.float64)
        reference_velocities.append(velocity.float().cpu())
        latents = (1 - nxt) * (latents - current * velocity) + nxt * noises[step].to(device=device, dtype=torch.float64)
        reference_latents.append(latents.detach().cpu())
    decode_and_save(pipe, helper, latents, output)
    reference = {
        "prompt_embeds": embeds_cpu,
        "initial_latent": reference_latents[0],
        "noises": noises,
        "inputs": reference_inputs,
        "velocities": reference_velocities,
        "latents": reference_latents,
        "times": [float(value) for value in times.detach().cpu()],
    }
    pipe.to("cpu")
    del pipe, embeds, velocity
    gc.collect(); torch.cuda.empty_cache()
    return reference, states, groups


def apply_restores(pipe: WanPipeline, layers: list[str], states: dict[str, dict[str, torch.Tensor]]) -> tuple[dict[str, Any], dict[str, int]]:
    before = untouched_fingerprint(pipe.transformer)
    checks = {}
    for name in layers:
        checks[name] = restore_one(pipe.transformer, name, states[name])
    integrity = validate_untouched(before, untouched_fingerprint(pipe.transformer))
    return checks, integrity


@torch.inference_mode()
def run_quant_case(
    case: str,
    layers: list[str],
    states: dict[str, dict[str, torch.Tensor]],
    reference: dict[str, Any],
    prompt: str,
    seed: int,
    height: int,
    width: int,
    frames: int,
    output_dir: Path,
    helper: Any,
    verify_only: bool,
) -> tuple[Path | None, dict[str, Any]]:
    del prompt, seed, height, width, frames  # Inputs are intentionally inherited from the paired BF16 cache.
    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    helper.load_quantized_transformer(pipe, CKPT, MODEL)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    checks, integrity = apply_restores(pipe, layers, states)
    meta: dict[str, Any] = {
        "method": "paired smooth-aware BF16 Q/K intervention",
        "case": case,
        "restored_layers": layers,
        "restored_count": len(layers),
        "checks": checks,
        "untouched_integrity": integrity,
        "checkpoint": str(CKPT),
    }
    if verify_only:
        pipe.to("cpu"); del pipe
        gc.collect(); torch.cuda.empty_cache()
        return None, meta

    embeds = reference["prompt_embeds"].to(device=device, dtype=pipe.transformer.dtype)
    times = torch.tensor(reference["times"], device=device, dtype=torch.float64)
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    latents = reference["initial_latent"].to(device=device, dtype=torch.float64).clone()
    per_step = []
    for step, (current, nxt) in enumerate(zip(times[:-1], times[1:], strict=True)):
        timestep = (current.float() * ones * 1000).to(dtype=pipe.transformer.dtype)
        teacher_input = reference["inputs"][step].to(device=device, dtype=torch.float64)
        teacher_velocity = pipe.transformer(hidden_states=teacher_input.to(pipe.transformer.dtype), timestep=timestep,
                                            encoder_hidden_states=embeds, return_dict=False)[0]
        teacher = tensor_metrics(teacher_velocity, reference["velocities"][step])
        if step == 0:
            rollout_velocity = teacher_velocity.to(torch.float64)
        else:
            rollout_velocity = pipe.transformer(hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                                                encoder_hidden_states=embeds, return_dict=False)[0].to(torch.float64)
        rollout = tensor_metrics(rollout_velocity, reference["velocities"][step])
        latents = (1 - nxt) * (latents - current * rollout_velocity) + nxt * reference["noises"][step].to(device=device, dtype=torch.float64)
        latent = tensor_metrics(latents, reference["latents"][step + 1])
        per_step.append({"case": case, "step": step, "timestep": float(current),
                         **{f"teacher_{key}": value for key, value in teacher.items()},
                         **{f"rollout_{key}": value for key, value in rollout.items()},
                         **{f"latent_{key}": value for key, value in latent.items()}})
        del teacher_input, teacher_velocity, rollout_velocity
    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    video = output_dir / f"{case}.mp4"
    decode_and_save(pipe, helper, latents, video)
    meta["video"] = str(video)
    meta["per_step"] = per_step
    pipe.to("cpu"); del pipe, embeds, latents
    gc.collect(); torch.cuda.empty_cache()
    return video, meta


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_per_step(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "per_step_metrics.json", {
        "definition": "teacher metrics compare quantized denoiser output on BF16 reference latent to BF16 output at the same timestep; rollout metrics compare the quantized model's own-trajectory output and updated latent to the corresponding BF16 trajectory. All tensors are compared online and not saved.",
        "rows": rows,
    })
    with (output_dir / "per_step_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_video_metrics(output_dir: Path, bf16_video: Path, videos: dict[str, Path]) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    rows = []
    for case in CASE_ORDER:
        video = videos[case]
        print(f"[video metrics] {case}", flush=True)
        row = {"case": case, "reference": str(bf16_video), "video": str(video)}
        row.update(compare_video(bf16_video, video, device))
        rows.append(row)
        torch.cuda.empty_cache()
    write_json(output_dir / "video_metrics.json", {
        "definition": "All metrics compare decoded RGB frames in [0,1] to the paired BF16 video. MSE/NMSE/MAE pool every frame and pixel; PSNR uses data range 1; SSIM and LPIPS-Alex are mean frame metrics.",
        "rows": rows,
    })
    with (output_dir / "video_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", choices=CASE_ORDER, default=list(CASE_ORDER))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    selected_cases = [case for case in CASE_ORDER if case in args.cases]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    helper = load_helper()

    bf16_video = output_dir / "bf16.mp4"
    reference, states, groups = build_bf16_reference(args.prompt, args.seed, args.height, args.width, args.frames,
                                                      bf16_video, helper)
    all_case_layers = {"nvfp4": [], **groups}
    metadata = {
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "schedule": "rCM TrigFlow->RectifiedFlow, 4 steps, guidance=0",
        "cases_requested": selected_cases,
        "bf16_video": str(bf16_video),
        "qk_group_counts": {name: len(layers) for name, layers in groups.items()},
        "stores_full_intermediate_tensors": False,
    }
    write_json(output_dir / "run_config.json", metadata)
    videos: dict[str, Path] = {}
    per_step_rows = []
    for case in selected_cases:
        print(f"[run] {case}: restoring {len(all_case_layers[case])} Q/K layers", flush=True)
        video, case_meta = run_quant_case(case, all_case_layers[case], states, reference, args.prompt, args.seed,
                                           args.height, args.width, args.frames, output_dir, helper, args.verify_only)
        write_json(output_dir / f"{case}.json", case_meta)
        if video is not None:
            videos[case] = video
            per_step_rows.extend(case_meta["per_step"])
    if args.verify_only:
        print(f"Verified requested cases -> {output_dir}", flush=True)
        return
    if set(selected_cases) != set(CASE_ORDER):
        raise RuntimeError("Video metrics require all four quantized cases; rerun without a restricted --cases list")
    write_per_step(output_dir, per_step_rows)
    write_video_metrics(output_dir, bf16_video, videos)
    print(f"Saved paired Q/K oracle experiment -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
