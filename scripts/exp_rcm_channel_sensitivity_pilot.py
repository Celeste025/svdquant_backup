#!/usr/bin/env python3
"""Single-prompt/single-timestep channel-sensitivity proxy validation.

The script differentiates the final rCM latent with respect to every quantized
Linear output at one denoising step.  Two Hutchinson quantities are accumulated
online:

  channel diag:sum_c mean_n E[(J^T r)_{n,c}^2] sum_n e_{n,c}^2
  directional: E[((J^T r)^T e)^2]

The first is a channel-diagonal approximation that averages token sensitivity;
the second retains all cross-coordinate terms for the observed error direction. Full
activations and gradients are never written to disk.  Quantization errors are
temporarily kept in CPU BF16 memory so all 300 layers share each trajectory
backward pass.
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
from diffusers import WanPipeline

from exp_rcm_trajectory_sensitivity_audit import (
    CACHE,
    CKPT,
    EPS,
    MODEL,
    CacheItem,
    as_tensor,
    block_index,
    json_default,
    layer_type,
    quantized_lowrank_linears,
    q_output,
    rcm_times,
    recovered_noises,
    sums,
)
from infer_rcm_wan_4step import load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT = ROOT / "results/reports/rcm_trajectory_sensitivity_audit/all300_2prompt_actual_normalized_1pct/trajectory_impact.csv"
DEFAULT_OUT = ROOT / "results/reports/rcm_channel_sensitivity_pilot/prompt0015_step0"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_prompt(cache_dir: Path, prompt_id: str) -> list[CacheItem]:
    found: list[CacheItem] = []
    for path in sorted(cache_dir.glob("*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        label = str(obj.get("filename", path.stem)).split("-")[0]
        if label == prompt_id:
            found.append(CacheItem(label, int(obj.get("step", -1)), path, obj))
    found.sort(key=lambda item: item.step)
    if [item.step for item in found] != [0, 1, 2, 3]:
        raise RuntimeError(f"Prompt {prompt_id!r} does not have exactly steps 0..3 in {cache_dir}")
    return found


def run_transformer_grad(model: torch.nn.Module, item: CacheItem, hidden: torch.Tensor) -> torch.Tensor:
    kw = item.payload["input_kwargs"]
    return as_tensor(
        model(
            hidden_states=hidden.to(dtype=torch.bfloat16),
            timestep=kw["timestep"].to(device=hidden.device, dtype=torch.bfloat16),
            encoder_hidden_states=kw["encoder_hidden_states"].to(device=hidden.device, dtype=torch.bfloat16),
            return_dict=False,
        )
    )


def differentiable_final(
    model: torch.nn.Module,
    items: list[CacheItem],
    start: int,
    noises: list[torch.Tensor],
    times: torch.Tensor,
    start_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = run_transformer_grad(model, items[start], start_hidden)
    next_time = times[start + 1]
    noise = noises[start] if start < 3 else torch.zeros_like(start_hidden, dtype=torch.float64)
    latent = (1 - next_time) * (start_hidden.to(torch.float64) - times[start] * velocity.to(torch.float64)) + next_time * noise
    for step in range(start + 1, 4):
        velocity_next = run_transformer_grad(model, items[step], latent)
        next_time = times[step + 1]
        noise = noises[step] if step < 3 else torch.zeros_like(latent)
        latent = (1 - next_time) * (latent - times[step] * velocity_next.to(torch.float64)) + next_time * noise
    return velocity, latent


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def read_exact(path: Path, prompt_id: str, step: int) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["prompt_id"]).zfill(len(prompt_id)) != prompt_id or int(row["step"]) != step:
                continue
            converted: dict[str, Any] = dict(row)
            for key, value in list(converted.items()):
                if key in {"layer", "layer_type", "mode", "role", "prompt_id"}:
                    continue
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    pass
            result[(row["layer"], row["mode"])] = converted
    if len(result) != 600:
        raise RuntimeError(f"Expected 600 exact rows for {prompt_id}/step{step}, found {len(result)}")
    return result


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            out[inverse == index] = out[inverse == index].mean()
    return out


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x, y = x[valid], y[valid]
    spearman = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
    pearson = float(np.corrcoef(x, y)[0, 1])
    log_pearson = float(np.corrcoef(np.log10(x + 1e-30), np.log10(y + 1e-30))[0, 1])
    return {"spearman": spearman, "pearson": pearson, "log10_pearson": log_pearson, "n": int(len(x))}


def top_overlap(pred: np.ndarray, exact: np.ndarray, k: int) -> int:
    return len(set(np.argsort(pred)[-k:]) & set(np.argsort(exact)[-k:]))


def analyse(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    metrics = {
        "local_nmse": "local NMSE",
        "mean_d_energy_pred_nmse": "mean(D) × total error",
        "diag_pred_nmse": "D(channel) × channel error",
        "directional_pred_nmse": "directional probe",
    }
    report: dict[str, Any] = {}
    for mode in ("actual", "normalized"):
        subset = [row for row in rows if row["mode"] == mode]
        exact = np.asarray([row["exact_final_latent_nmse"] for row in subset], dtype=np.float64)
        report[mode] = {}
        for key, label in metrics.items():
            pred = np.asarray([row[key] for row in subset], dtype=np.float64)
            entry = correlation(pred, exact)
            entry["top_overlap"] = {str(k): top_overlap(pred, exact, k) for k in (10, 20, 30)}
            report[mode][key] = entry

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for axis, (key, label) in zip(axes.flat, metrics.items()):
            pred = np.asarray([row[key] for row in subset], dtype=np.float64)
            axis.scatter(pred, exact, s=13, alpha=.65)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(label)
            axis.set_ylabel("exact final-latent NMSE")
            axis.set_title(f"Spearman={report[mode][key]['spearman']:.3f}")
            axis.grid(alpha=.2)
        fig.suptitle(f"prompt={subset[0]['prompt_id']} step={int(subset[0]['step'])} mode={mode}")
        fig.tight_layout()
        fig.savefig(out / f"proxy_vs_exact_{mode}.png", dpi=180)
        plt.close(fig)

        ordered = sorted(subset, key=lambda row: row["exact_final_latent_nmse"], reverse=True)[:30]
        x = np.arange(len(ordered))
        fig, axis = plt.subplots(figsize=(14, 5))
        exact_top = np.asarray([row["exact_final_latent_nmse"] for row in ordered])
        axis.plot(x, exact_top / max(exact_top.max(), EPS), "ko-", ms=3, label="exact")
        for key, label in list(metrics.items())[1:]:
            values = np.asarray([row[key] for row in ordered])
            axis.plot(x, values / max(values.max(), EPS), "o-", ms=3, label=label)
        axis.set_xticks(x)
        axis.set_xticklabels([row["layer"].replace("blocks.", "b") for row in ordered], rotation=75, ha="right", fontsize=7)
        axis.set_ylabel("score normalized by displayed maximum")
        axis.set_title(f"Exact top-30 comparison ({mode})")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"exact_top30_proxy_comparison_{mode}.png", dpi=180)
        plt.close(fig)
    (out / "correlations.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", default="0015")
    parser.add_argument("--step", type=int, default=0, choices=range(4))
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=0, help="Smoke-test prefix; 0 means all 300 layers")
    parser.add_argument("--probe-seed", type=int, default=20260903)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--exact-csv", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.probes < 1:
        raise ValueError("--probes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "endpoint": "final_latent",
        "all_tokens": True,
        "temporary_error_cache": "CPU BF16; never written",
        "probe_distribution": "Rademacher",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=json_default))

    device = torch.device("cuda")
    items = load_prompt(args.cache_dir, args.prompt_id)
    exact = read_exact(args.exact_csv, args.prompt_id, args.step)
    times = rcm_times(device)
    noises = recovered_noises(items, times, device)

    pipe_bf = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe_bf.text_encoder.to("cpu")
    pipe_bf.vae.to("cpu")
    # LowRankBranch instances live on hooks instead of in the transformer's
    # registered module tree. Create them only after the base pipeline is on
    # CUDA, matching the established quantized-loader ordering.
    pipe_q = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe_q.text_encoder.to("cpu")
    pipe_q.vae.to("cpu")
    load_quantized_transformer(pipe_q, args.checkpoint, MODEL)
    bf, quant = pipe_bf.transformer, pipe_q.transformer
    freeze(bf)
    freeze(quant)
    bf.enable_gradient_checkpointing()
    layers = quantized_lowrank_linears(quant, bf)
    if args.max_layers:
        layers = layers[: args.max_layers]
    bfmods = dict(bf.named_modules())

    # CPU BF16 error directions are large (~34.5 GiB for this configuration)
    # but avoid either 300 trajectory forwards or an unsafe GPU resident cache.
    errors: dict[str, torch.Tensor] = {}
    error_energy_channel: dict[str, torch.Tensor] = {}
    ref_energy: dict[str, float] = {}
    capture_forward = {"enabled": True}
    # Some Linear inputs (notably cross-attention K/V from frozen text
    # embeddings) otherwise require no gradients. A shared zero-valued anchor
    # makes every target output differentiable without changing its value.
    gradient_anchor = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
    handles = []

    def make_forward_hook(name: str):
        def hook(_module: torch.nn.Module, call_args: tuple[Any, ...], output: Any) -> None:
            original = as_tensor(output)
            if capture_forward["enabled"] and name not in errors:
                ref = original.detach()
                with torch.no_grad():
                    quant_out = q_output(quant, name, call_args[0].detach())
                    error = (quant_out - ref).to("cpu", dtype=torch.bfloat16)
                    dims = tuple(range(error.ndim - 1))
                    energy = torch.sum(error * error, dim=dims, dtype=torch.float32)
                    reference_energy = float(torch.sum(ref * ref, dtype=torch.float32))
                errors[name] = error
                error_energy_channel[name] = energy
                ref_energy[name] = reference_energy
            return original + gradient_anchor.to(dtype=original.dtype) * 0
        return hook

    # Full backward hooks survive activation recomputation.  A hook on the
    # selected-step denoiser output enables them only after future steps have
    # already backpropagated, so repeated module invocations are not mixed.
    backward_enabled = {"value": False}
    sum_g2_channel: dict[str, torch.Tensor] = {}
    directional: dict[str, list[float]] = defaultdict(list)

    def make_backward_hook(name: str):
        def hook(_module: torch.nn.Module, _grad_input: Any, grad_output: Any) -> None:
            if not backward_enabled["value"]:
                return
            grad = as_tensor(grad_output).detach()
            dims = tuple(range(grad.ndim - 1))
            g2 = torch.sum(grad * grad, dim=dims, dtype=torch.float32).cpu()
            sum_g2_channel[name] = sum_g2_channel.get(name, torch.zeros_like(g2)) + g2
            error_gpu = errors[name].to(device=grad.device)
            dot = torch.sum(grad * error_gpu, dtype=torch.float32)
            directional[name].append(float(dot))
            del error_gpu
        return hook

    for name in layers:
        handles.append(bfmods[name].register_forward_hook(make_forward_hook(name)))
        handles.append(bfmods[name].register_full_backward_hook(make_backward_hook(name)))

    start_hidden = items[args.step].hidden.to(device=device, dtype=torch.bfloat16).detach().requires_grad_(True)
    start_velocity, final_latent = differentiable_final(bf, items, args.step, noises, times, start_hidden)
    capture_forward["enabled"] = False
    if len(errors) != len(layers):
        raise RuntimeError(f"Expected {len(layers)} error directions, captured {len(errors)}")

    reference_final = (
        items[3].hidden.to(device=device, dtype=torch.float64)
        - times[3] * items[3].velocity.to(device=device, dtype=torch.float64)
    )
    zero_e2, zero_r2, _ = sums(final_latent.detach(), reference_final)
    zero_nmse = zero_e2 / max(zero_r2, EPS)
    if zero_nmse >= 5e-5:
        raise RuntimeError(f"Differentiable zero-injection trajectory mismatch: NMSE={zero_nmse:.3e}")

    # The selected-step output hook marks the boundary during reverse traversal:
    # future denoising calls are ignored; only this timestep's Linear gradients
    # contribute to D and g^T e.
    def enable_selected_backward(gradient: torch.Tensor) -> torch.Tensor:
        backward_enabled["value"] = True
        return gradient

    start_velocity.register_hook(enable_selected_backward)
    generator = torch.Generator(device=device).manual_seed(args.probe_seed)
    for probe_index in range(args.probes):
        backward_enabled["value"] = False
        probe = torch.empty_like(final_latent, dtype=torch.int8).random_(0, 2, generator=generator)
        probe = probe.to(torch.float64).mul_(2).sub_(1)
        scalar = torch.sum(final_latent * probe)
        torch.autograd.grad(scalar, (start_hidden, gradient_anchor), retain_graph=probe_index + 1 < args.probes)
        if len(directional) != len(layers) or any(len(values) != probe_index + 1 for values in directional.values()):
            raise RuntimeError(f"Probe {probe_index}: incomplete selected-step gradients ({len(directional)}/{len(layers)} layers)")
        print(f"probe {probe_index + 1}/{args.probes} complete; cuda_peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
        del probe, scalar

    for handle in handles:
        handle.remove()

    final_ref2 = float(torch.sum(reference_final * reference_final))
    rows: list[dict[str, Any]] = []
    channel_payload: dict[str, Any] = {}
    max_local_relerr = 0.0
    for name in layers:
        d_channel = sum_g2_channel[name] / args.probes
        e_channel = error_energy_channel[name]
        actual_err2 = float(e_channel.sum())
        exact_actual = exact[(name, "actual")]
        exact_normal = exact[(name, "normalized")]
        relerr = abs(actual_err2 - float(exact_actual["local_err2"])) / max(float(exact_actual["local_err2"]), EPS)
        max_local_relerr = max(max_local_relerr, relerr)
        if relerr > .01:
            raise RuntimeError(f"{name}: captured local err2 differs from exact by {relerr:.2%}")
        mean_d = float(d_channel.sum()) / int(exact_actual["local_numel"])
        # D_c is the token-average diagonal sensitivity for channel c. Both
        # accumulated tensors above are token sums, so divide the gradient
        # energy by positions/channel before combining with error energy.
        positions_per_channel = int(exact_actual["local_numel"]) / d_channel.numel()
        diag_actual_err2 = float(torch.dot(d_channel / positions_per_channel, e_channel))
        directional_actual_err2 = float(np.mean(np.square(directional[name], dtype=np.float64)))
        for mode, exact_row in (("actual", exact_actual), ("normalized", exact_normal)):
            scale2 = 1.0 if mode == "actual" else float(exact_normal["scale"]) ** 2
            mode_err2 = actual_err2 * scale2
            rows.append({
                "prompt_id": args.prompt_id,
                "step": args.step,
                "layer": name,
                "layer_type": layer_type(name),
                "block": block_index(name),
                "mode": mode,
                "local_nmse": float(exact_row["local_nmse"]),
                "exact_final_latent_nmse": float(exact_row["final_latent_nmse"]),
                "mean_d": mean_d,
                "mean_d_energy_pred_nmse": mean_d * mode_err2 / final_ref2,
                "diag_pred_nmse": diag_actual_err2 * scale2 / final_ref2,
                "directional_pred_nmse": directional_actual_err2 * scale2 / final_ref2,
                "scale": float(exact_row["scale"]),
            })
        total_error = max(float(e_channel.sum()), EPS)
        channel_payload[name] = {
            "d": d_channel.tolist(),
            "error_energy_fraction": (e_channel / total_error).tolist(),
            "output_channels": int(d_channel.numel()),
        }

    write_csv(args.output_dir / "per_layer_proxy.csv", rows)
    (args.output_dir / "per_layer_proxy.json").write_text(json.dumps(rows, indent=2))
    torch.save(channel_payload, args.output_dir / "channel_statistics.pt")
    report = analyse(rows, args.output_dir)
    audit = {
        "zero_injection_final_nmse": zero_nmse,
        "max_local_err2_relative_difference_vs_exact": max_local_relerr,
        "captured_layers": len(errors),
        "probes": args.probes,
        "cuda_peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "temporary_cpu_error_cache_GiB": sum(value.numel() * value.element_size() for value in errors.values()) / 2**30,
    }
    (args.output_dir / "correctness.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"correctness": audit, "correlations": report}, indent=2), flush=True)

    del errors, error_energy_channel, sum_g2_channel, directional
    pipe_bf.to("cpu")
    pipe_q.to("cpu")
    del pipe_bf, pipe_q, final_latent, start_velocity, start_hidden
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
