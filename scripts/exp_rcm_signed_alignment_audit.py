#!/usr/bin/env python3
"""All-layer signed correction-alignment audit for rCM-Wan W4A4.

For every quantized Linear and denoising timestep, measure the first-order
change in final-latent NMSE obtained by moving that Linear's output from its
current W4A4 value toward its smooth-aware BF16 value.  One backward pass per
timestep yields all 300 layer scores; no activation or output tensor is saved.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import WanPipeline

from exp_rcm_restore_top_fraction_bf16 import input_smoother
from exp_rcm_trajectory_sensitivity_audit import (
    CKPT,
    EPS,
    MODEL,
    as_tensor,
    block_index,
    json_default,
    layer_type,
    quantized_lowrank_linears,
    sums,
)
from infer_rcm_wan_4step import load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/reports/rcm_signed_alignment_audit/airplane_seed303"
DEFAULT_UNSIGNED = ROOT / "results/reports/rcm_trajectory_sensitivity_audit/all300_2prompt_actual_normalized_1pct/trajectory_impact.csv"
PROMPT = "An airplane soaring through a clear blue sky above white clouds."


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def make_autograd_safe(module: torch.nn.Module) -> dict[str, int]:
    """Clone PTQ tensors created under inference_mode without changing values."""
    counts = {"parameters": 0, "buffers": 0, "hook_tensors": 0, "branch_parameters": 0}
    seen_modules: set[int] = set()

    def clone_module(current: torch.nn.Module, branch: bool = False) -> None:
        if id(current) in seen_modules:
            return
        seen_modules.add(id(current))
        for child in current.modules():
            if id(child) in seen_modules and child is not current:
                continue
            seen_modules.add(id(child))
            for name, parameter in list(child._parameters.items()):
                if parameter is not None and parameter.is_inference():
                    with torch.inference_mode(False):
                        replacement = torch.nn.Parameter(parameter.detach().clone(), requires_grad=False)
                    setattr(child, name, replacement)
                    counts["branch_parameters" if branch else "parameters"] += 1
            for name, buffer in list(child._buffers.items()):
                if buffer is not None and buffer.is_inference():
                    with torch.inference_mode(False):
                        replacement = buffer.detach().clone()
                    setattr(child, name, replacement)
                    counts["buffers"] += 1

    clone_module(module)
    seen_objects: set[int] = set()

    def clone_object_tensors(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 3 or id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            clone_module(obj, branch=True)
            return
        if not hasattr(obj, "__dict__"):
            return
        for name, value in list(vars(obj).items()):
            if torch.is_tensor(value) and value.is_inference():
                with torch.inference_mode(False):
                    setattr(obj, name, value.detach().clone())
                counts["hook_tensors"] += 1
            elif isinstance(value, (list, tuple)):
                for item in value:
                    clone_object_tensors(item, depth + 1)
            elif isinstance(value, dict):
                for item in value.values():
                    clone_object_tensors(item, depth + 1)
            elif hasattr(value, "__dict__"):
                clone_object_tensors(value, depth + 1)

    for current in module.modules():
        for hook in list(current._forward_pre_hooks.values()) + list(current._forward_hooks.values()):
            clone_object_tensors(getattr(hook, "processor", None))
            clone_object_tensors(getattr(hook, "branch", None))
    return counts


def rcm_schedule(device: torch.device, sigma_max: float = 80.0) -> torch.Tensor:
    trig = torch.tensor([math.atan(sigma_max), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    return torch.sin(trig) / (torch.cos(trig) + torch.sin(trig))


def transformer_forward(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    return as_tensor(model(
        hidden_states=hidden.to(dtype=torch.bfloat16),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    ))


def update_latent(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    return (1 - t_next) * (latent.to(torch.float64) - t_cur * velocity.to(torch.float64)) + t_next * noise


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    noises: list[torch.Tensor],
    times: torch.Tensor,
    timesteps: list[torch.Tensor],
    prompt_embeds: torch.Tensor,
    keep_starts: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    latent = initial
    starts: list[torch.Tensor] = []
    for step in range(4):
        if keep_starts:
            starts.append(latent.detach().cpu())
        velocity = transformer_forward(model, latent, timesteps[step], prompt_embeds)
        latent = update_latent(latent, velocity, times[step], times[step + 1], noises[step])
    return latent, starts


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            result[inverse == index] = result[inverse == index].mean()
    return result


def spearman(x: list[float], y: list[float]) -> float:
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.corrcoef(ranks(xa), ranks(ya))[0, 1])


def read_unsigned(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "actual":
                values[row["layer"]].append(float(row["final_latent_nmse"]))
    return {name: float(np.mean(items)) for name, items in values.items()}


def make_plots(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], out: Path) -> None:
    layer_names = [row["layer"] for row in summary_rows]
    matrix = np.zeros((len(layer_names), 4), dtype=np.float64)
    lookup = {(row["layer"], int(row["step"])): float(row["predicted_nmse_reduction"]) for row in rows}
    for i, name in enumerate(layer_names):
        for step in range(4):
            matrix[i, step] = lookup.get((name, step), np.nan)
    bound = float(np.nanpercentile(np.abs(matrix), 98)) or 1.0
    fig, axis = plt.subplots(figsize=(7, 13))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    axis.set_xticks(range(4), [f"step {i}" for i in range(4)])
    axis.set_ylabel("layers, sorted by aggregate signed benefit")
    axis.set_title("Predicted final-latent NMSE reduction\nred = beneficial, blue = harmful")
    fig.colorbar(image, ax=axis, label="first-order NMSE reduction")
    fig.tight_layout()
    fig.savefig(out / "layer_by_timestep_signed_benefit.png", dpi=180)
    plt.close(fig)

    values = np.asarray([row["predicted_nmse_reduction_sum"] for row in summary_rows])
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=50, color="tab:red" if values.mean() >= 0 else "tab:blue", alpha=.8)
    axis.axvline(0, color="black", lw=1)
    axis.set_xlabel("sum over four steps: predicted final-latent NMSE reduction")
    axis.set_ylabel("layers")
    axis.set_title("Signed correction alignment")
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "signed_benefit_histogram.png", dpi=180)
    plt.close(fig)

    if "old_unsigned_actual_score" in summary_rows[0]:
        unsigned = np.asarray([row["old_unsigned_actual_score"] for row in summary_rows])
        fig, axis = plt.subplots(figsize=(7, 6))
        colors = np.where(values >= 0, "tab:red", "tab:blue")
        axis.scatter(unsigned, np.abs(values), c=colors, s=18, alpha=.7)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("old unsigned single-layer final impact")
        axis.set_ylabel("absolute signed first-order benefit")
        axis.set_title("Unsigned impact does not determine correction sign")
        axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / "old_unsigned_vs_signed.png", dpi=180)
        plt.close(fig)


def analyse(rows: list[dict[str, Any]], unsigned_path: Path, out: Path) -> dict[str, Any]:
    for row in rows:
        row["layer_type"] = layer_type(row["layer"])
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)
    unsigned = read_unsigned(unsigned_path) if unsigned_path.exists() else {}
    summary_rows: list[dict[str, Any]] = []
    for name, items in by_layer.items():
        items.sort(key=lambda row: int(row["step"]))
        benefit = sum(float(row["predicted_nmse_reduction"]) for row in items)
        correction2 = sum(float(row["correction_err2"]) for row in items)
        entry: dict[str, Any] = {
            "layer": name,
            "layer_type": layer_type(name),
            "block": block_index(name),
            "predicted_nmse_reduction_sum": benefit,
            "beneficial_steps": sum(float(row["predicted_nmse_reduction"]) > 0 for row in items),
            "correction_err2_sum": correction2,
            "aggregate_cosine": -sum(float(row["grad_dot_correction"]) for row in items)
                / max(math.sqrt(sum(float(row["grad_err2"]) for row in items) * correction2), EPS),
        }
        if name in unsigned:
            entry["old_unsigned_actual_score"] = unsigned[name]
        summary_rows.append(entry)
    summary_rows.sort(key=lambda row: (-float(row["predicted_nmse_reduction_sum"]), row["layer"]))
    write_csv(out / "per_layer_signed_summary.csv", summary_rows)
    (out / "per_layer_signed_summary.json").write_text(json.dumps(summary_rows, indent=2))

    positive = [row for row in summary_rows if float(row["predicted_nmse_reduction_sum"]) > 0]
    old_top30 = sorted(summary_rows, key=lambda row: float(row.get("old_unsigned_actual_score", -math.inf)), reverse=True)[:30]
    signed_top30 = summary_rows[:30]
    report: dict[str, Any] = {
        "layers": len(summary_rows),
        "aggregate_positive_layers": len(positive),
        "aggregate_negative_layers": len(summary_rows) - len(positive),
        "old_unsigned_top30_positive": sum(float(row["predicted_nmse_reduction_sum"]) > 0 for row in old_top30),
        "old_unsigned_top30_negative": sum(float(row["predicted_nmse_reduction_sum"]) <= 0 for row in old_top30),
        "old_unsigned_top30_predicted_nmse_reduction_sum": sum(float(row["predicted_nmse_reduction_sum"]) for row in old_top30),
        "signed_top30_predicted_nmse_reduction_sum": sum(float(row["predicted_nmse_reduction_sum"]) for row in signed_top30),
        "top30_overlap_old_unsigned_vs_signed": len({row["layer"] for row in old_top30} & {row["layer"] for row in signed_top30}),
        "per_step": {},
        "per_layer_type": {},
        "signed_top30": [row["layer"] for row in signed_top30],
        "signed_bottom30": [row["layer"] for row in summary_rows[-30:]],
        "old_unsigned_top30": [row["layer"] for row in old_top30],
    }
    for step in sorted({int(row["step"]) for row in rows}):
        step_values = [float(row["predicted_nmse_reduction"]) for row in rows if int(row["step"]) == step]
        report["per_step"][str(step)] = {
            "positive_layers": sum(value > 0 for value in step_values),
            "negative_layers": sum(value <= 0 for value in step_values),
            "sum_predicted_nmse_reduction": sum(step_values),
            "median_predicted_nmse_reduction": float(np.median(step_values)),
        }
    types = sorted({row["layer_type"] for row in summary_rows})
    for kind in types:
        type_values = [float(row["predicted_nmse_reduction_sum"]) for row in summary_rows if row["layer_type"] == kind]
        report["per_layer_type"][kind] = {
            "layers": len(type_values),
            "positive_layers": sum(value > 0 for value in type_values),
            "sum_predicted_nmse_reduction": sum(type_values),
            "median_predicted_nmse_reduction": float(np.median(type_values)),
        }
    if unsigned:
        report["spearman_old_unsigned_vs_signed_benefit"] = spearman(
            [float(row["old_unsigned_actual_score"]) for row in summary_rows],
            [float(row["predicted_nmse_reduction_sum"]) for row in summary_rows],
        )
    (out / "analysis.json").write_text(json.dumps(report, indent=2))
    make_plots(rows, summary_rows, out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unsigned-csv", type=Path, default=DEFAULT_UNSIGNED)
    parser.add_argument("--max-layers", type=int, default=0, help="prefix smoke test; 0 audits all 300")
    parser.add_argument("--audit-steps", type=int, nargs="+", default=[0, 1, 2, 3], choices=range(4))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args) | {
        "score": "-2 * <d(half final NMSE)/d(y_quant), y_bf16_restore-y_quant>",
        "interpretation": "positive predicts NMSE reduction; negative predicts harmful cancellation removal",
        "all_tokens": True,
        "saved_tensors": "scalar statistics only",
    }, indent=2, default=json_default))

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    pipe_bf = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe_q = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    load_quantized_transformer(pipe_q, args.checkpoint, MODEL)
    bf, quant = pipe_bf.transformer, pipe_q.transformer
    safe_counts = make_autograd_safe(quant)
    print(f"autograd-safe tensor clones: {safe_counts}", flush=True)
    freeze(bf)
    freeze(quant)
    quant.enable_gradient_checkpointing()
    layers = quantized_lowrank_linears(quant, bf)
    if args.max_layers:
        layers = layers[:args.max_layers]
    bfmods = dict(bf.named_modules())
    qmods = dict(quant.named_modules())
    bf_states = {
        name: {
            "weight": bfmods[name].weight.detach().clone(),
            **({"bias": bfmods[name].bias.detach().clone()} if bfmods[name].bias is not None else {}),
        }
        for name in layers
    }

    prompt_embeds, _ = pipe_bf.encode_prompt(
        prompt=args.prompt,
        do_classifier_free_guidance=False,
        max_sequence_length=512,
        device=device,
        dtype=bf.dtype,
    )
    pipe_bf.text_encoder.to("cpu")
    pipe_q.text_encoder.to("cpu")
    pipe_bf.vae.to("cpu")
    pipe_q.vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = pipe_bf.prepare_latents(
        batch_size=1,
        num_channels_latents=bf.config.in_channels,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    times = rcm_schedule(device)
    initial = initial.to(torch.float64) * times[0]
    noises = [
        torch.randn(initial.shape, dtype=torch.float32, device=device, generator=generator)
        if step < 3 else torch.zeros_like(initial, dtype=torch.float64)
        for step in range(4)
    ]
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    timesteps = [(times[step].float() * ones * 1000).to(torch.bfloat16) for step in range(4)]

    bf_final, _ = rollout(bf, initial, noises, times, timesteps, prompt_embeds)
    q_final, q_starts = rollout(quant, initial, noises, times, timesteps, prompt_embeds, keep_starts=True)
    baseline_e2, baseline_r2, baseline_n = sums(q_final, bf_final)
    baseline_nmse = baseline_e2 / max(baseline_r2, EPS)
    print(f"baseline quant-vs-BF16 final NMSE={baseline_nmse:.8g}", flush=True)
    pipe_bf.transformer.to("cpu")
    del bf, bfmods, pipe_bf
    gc.collect()
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    smooth_metadata: dict[str, dict[str, Any]] = {}
    for name in layers:
        smoother, direct = input_smoother(quant, name, qmods[name])
        smooth_metadata[name] = {
            "owner": "direct" if direct else "parent" if smoother is not None else "none",
            "upscale": bool(getattr(smoother, "upscale", False)),
        }

    for selected_step in args.audit_steps:
        corrections: dict[str, torch.Tensor] = {}
        correction_err2: dict[str, float] = {}
        raw_inputs: dict[str, torch.Tensor] = {}
        grad_dot: dict[str, float] = {}
        grad_err2: dict[str, float] = {}
        backward_enabled = {"value": False}
        anchor = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        handles = []

        def make_pre_hook(name: str):
            def hook(_module: torch.nn.Module, call_args: tuple[Any, ...]) -> None:
                if name not in corrections:
                    raw_inputs[name] = call_args[0].detach()
            return hook

        def make_forward_hook(name: str):
            def hook(module: torch.nn.Module, _call_args: tuple[Any, ...], output: Any) -> torch.Tensor:
                quant_output = as_tensor(output)
                if name not in corrections:
                    raw = raw_inputs.pop(name)
                    state = bf_states[name]
                    smoother, direct = input_smoother(quant, name, module)
                    weight = state["weight"]
                    bf_input = raw
                    if smoother is not None and not direct:
                        scale = smoother.smooth_scale.to(device=weight.device, dtype=weight.dtype).view(1, -1)
                        weight = weight * (scale.reciprocal() if bool(getattr(smoother, "upscale", False)) else scale)
                    bias = state.get("bias")
                    with torch.no_grad():
                        restored = F.linear(bf_input, weight, bias)
                        correction = (restored - quant_output.detach()).to("cpu", dtype=torch.bfloat16)
                        correction_err2[name] = float(torch.sum(correction * correction, dtype=torch.float32))
                    corrections[name] = correction
                anchored = quant_output + anchor.to(dtype=quant_output.dtype) * 0

                # Tensor hooks also cover cross-attention K/V whose frozen text
                # inputs do not make nn.Module full-backward hooks fire reliably.
                def capture_gradient(grad: torch.Tensor) -> torch.Tensor:
                    if backward_enabled["value"] and name not in grad_dot:
                        correction = corrections[name].to(device=grad.device)
                        grad_dot[name] = float(torch.sum(grad.detach() * correction, dtype=torch.float32))
                        grad_err2[name] = float(torch.sum(grad.detach() * grad.detach(), dtype=torch.float32))
                        del correction
                    return grad

                anchored.register_hook(capture_gradient)
                return anchored
            return hook

        for name in layers:
            module = qmods[name]
            handles.append(module.register_forward_pre_hook(make_pre_hook(name), prepend=True))
            handles.append(module.register_forward_hook(make_forward_hook(name)))

        latent = q_starts[selected_step].to(device=device, dtype=torch.float64).detach().requires_grad_(True)
        selected_velocity = transformer_forward(quant, latent, timesteps[selected_step], prompt_embeds)

        def enable_backward(gradient: torch.Tensor) -> torch.Tensor:
            backward_enabled["value"] = True
            return gradient

        selected_velocity.register_hook(enable_backward)
        current = update_latent(latent, selected_velocity, times[selected_step], times[selected_step + 1], noises[selected_step])
        for step in range(selected_step + 1, 4):
            velocity = transformer_forward(quant, current, timesteps[step], prompt_embeds)
            current = update_latent(current, velocity, times[step], times[step + 1], noises[step])
        replay_e2, _, _ = sums(current.detach(), q_final)
        replay_nmse = replay_e2 / max(baseline_e2 + float(torch.sum(q_final.float().square())), EPS)
        if replay_nmse >= 5e-5:
            raise RuntimeError(f"step {selected_step}: zero-change quant replay mismatch NMSE={replay_nmse:.3e}")
        loss = .5 * torch.sum((current - bf_final).square()) / torch.sum(bf_final.square()).clamp_min(EPS)
        torch.autograd.grad(loss, (anchor, latent), allow_unused=True)
        if len(corrections) != len(layers) or len(grad_dot) != len(layers):
            raise RuntimeError(
                f"step {selected_step}: incomplete hooks: correction={len(corrections)}, gradient={len(grad_dot)}, expected={len(layers)}"
            )
        for handle in handles:
            handle.remove()
        for name in layers:
            dot = grad_dot[name]
            ce2 = correction_err2[name]
            ge2 = grad_err2[name]
            rows.append({
                "step": selected_step,
                "layer": name,
                "layer_type": layer_type(name),
                "block": block_index(name),
                "smooth_owner": smooth_metadata[name]["owner"],
                "smooth_upscale": smooth_metadata[name]["upscale"],
                "grad_dot_correction": dot,
                "predicted_half_nmse_change": dot,
                "predicted_nmse_change": 2 * dot,
                "predicted_nmse_reduction": -2 * dot,
                "correction_err2": ce2,
                "grad_err2": ge2,
                "correction_gradient_cosine": dot / max(math.sqrt(ce2 * ge2), EPS),
            })
        print(
            f"step {selected_step}/3 complete: layers={len(layers)}, "
            f"beneficial={sum(grad_dot[name] < 0 for name in layers)}, "
            f"cpu_correction_GiB={sum(t.numel()*t.element_size() for t in corrections.values())/2**30:.2f}, "
            f"cuda_peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}",
            flush=True,
        )
        del corrections, correction_err2, raw_inputs, grad_dot, grad_err2, anchor, latent, selected_velocity, current, loss
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "per_layer_per_step_signed_alignment.csv", rows)
    (args.output_dir / "per_layer_per_step_signed_alignment.json").write_text(json.dumps(rows, indent=2))
    report = analyse(rows, args.unsigned_csv, args.output_dir)
    correctness = {
        "target_layers": len(layers),
        "audited_steps": args.audit_steps,
        "baseline_final_nmse": baseline_nmse,
        "baseline_final_mse": baseline_e2 / baseline_n,
        "zero_change_replay_threshold": 5e-5,
        "cuda_peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "correctness.json").write_text(json.dumps(correctness, indent=2))
    print(json.dumps({"correctness": correctness, "analysis": report}, indent=2), flush=True)

    pipe_q.to("cpu")
    del pipe_q, quant, qmods, bf_states, q_starts, q_final, bf_final
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
