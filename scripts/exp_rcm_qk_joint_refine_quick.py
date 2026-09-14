#!/usr/bin/env python3
"""Fail-fast Q/K low-rank refinement on the original rCM-Wan calibration caches.

This experiment keeps the quantized base weights, smoothing, activation QDQ,
rank, and branch topology fixed.  Only the existing Q/K LowRankBranch B
matrices are updated.  Two controls are compared from the identical checkpoint:

* independent: minimize separate Q and K projection NMSE;
* coupled: minimize full attention-output NMSE with Q and K updated jointly.

The full attention objective intentionally matches the narrow remaining gap to
SVDQuant: SVDQuant scores SVD candidates with OutputsError, whereas this script
directly differentiates that error into the paired Q/K branch parameters.
No captured activation, Q/K tensor, or attention output is written to disk.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import WanPipeline


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/models/svdquant-wjq")
MODEL = DATA / "models" / "rcm-Wan2.1-T2V-1.3B-Diffusers"
CKPT = DATA / "ckpts" / "rcm-wan2.1-1.3b-real-nvfp4-s16"
CACHE_DIR = (
    DATA
    / "datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches"
)
DEFAULT_OUT = ROOT / "results" / "qk_coupled_rcm" / "quick_4pairs"
PAIR_NAMES = (
    "blocks.25.attn1",
    "blocks.26.attn1",
    "blocks.15.attn2",
    "blocks.16.attn2",
)


def load_helper():
    path = ROOT / "scripts/infer_rcm_wan_4step.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def tensor_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Expected tensor-like output, got {type(value)}")


def nmse(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    error = got.float() - ref.float()
    return error.square().sum() / ref.float().square().sum().clamp_min(1e-12)


def pooled_update(total: list[float], got: torch.Tensor, ref: torch.Tensor) -> None:
    error = got.float() - ref.float()
    total[0] += float(error.square().sum().item())
    total[1] += float(ref.float().square().sum().item())


def cache_paths(num_prompts: int) -> list[Path]:
    paths = []
    for prompt in range(num_prompts):
        for step in range(4):
            path = CACHE_DIR / f"{prompt:04d}-{step:05d}-0.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


@torch.inference_mode()
def collect_bf16_records(
    paths: list[Path], pair_names: tuple[str, ...], device: torch.device
) -> dict[str, list[dict[str, Any]]]:
    """Capture only selected attention inputs/targets in CPU RAM."""
    print(f"[collect] loading BF16 model; samples={len(paths)} pairs={len(pair_names)}", flush=True)
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    modules = dict(pipe.transformer.named_modules())
    current: dict[str, dict[str, Any]] = {}
    records = {name: [] for name in pair_names}
    handles = []

    for name in pair_names:
        attn = modules[name]

        def attn_pre(_module, args, kwargs, *, pair=name):
            del args
            current[pair] = {"kwargs": to_device(kwargs, "cpu")}

        def q_post(_module, _args, output, *, pair=name):
            current[pair]["q_ref"] = tensor_output(output).detach().cpu()

        def k_post(_module, _args, output, *, pair=name):
            current[pair]["k_ref"] = tensor_output(output).detach().cpu()

        def attn_post(_module, _args, _kwargs, output, *, pair=name):
            current[pair]["attn_ref"] = tensor_output(output).detach().cpu()

        handles.append(attn.register_forward_pre_hook(attn_pre, with_kwargs=True))
        handles.append(attn.to_q.register_forward_hook(q_post))
        handles.append(attn.to_k.register_forward_hook(k_post))
        handles.append(attn.register_forward_hook(attn_post, with_kwargs=True))

    try:
        for index, path in enumerate(paths):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            args = to_device(payload["input_args"], device)
            kwargs = to_device(payload["input_kwargs"], device)
            kwargs["return_dict"] = False
            current.clear()
            pipe.transformer(*args, **kwargs)
            if set(current) != set(pair_names):
                raise RuntimeError(f"capture mismatch: {sorted(current)}")
            for name in pair_names:
                row = current[name]
                required = {"kwargs", "q_ref", "k_ref", "attn_ref"}
                if set(row) != required:
                    raise RuntimeError(f"{name}: captured {sorted(row)}, expected {sorted(required)}")
                row["cache"] = path.name
                records[name].append(row)
            print(f"[collect] {index + 1}/{len(paths)} {path.name}", flush=True)
            del payload, args, kwargs
    finally:
        for handle in handles:
            handle.remove()
        pipe.to("cpu")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return records


def low_rank_branch(module: torch.nn.Module):
    matches = []
    for hook in module._forward_hooks.values():
        branch = getattr(hook, "branch", None)
        if type(branch).__name__ == "LowRankBranch":
            matches.append(branch)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one LowRankBranch on {module}, found {len(matches)}")
    return matches[0]


def materialize_frozen_params_for_backward(module: torch.nn.Module) -> None:
    """Replace inference-mode parameters by equal frozen normal Parameters.

    The PTQ loader intentionally runs under inference mode.  A coupled loss
    backpropagates through q/k norms and the frozen output projection, whose
    weights autograd must save to form the gradient for the Q/K branches.
    """
    # Low-rank branches live inside hook objects, so they are not returned by
    # Module.modules(). Include them explicitly as they sit on the Q/K/V/O
    # projection paths used by a full attention forward.
    children = list(module.modules())
    for child in list(children):
        for hooks in (child._forward_pre_hooks, child._forward_hooks):
            for hook in hooks.values():
                branch = getattr(hook, "branch", None)
                if isinstance(branch, torch.nn.Module):
                    children.extend(branch.modules())
    seen: set[int] = set()
    for child in children:
        if id(child) in seen:
            continue
        seen.add(id(child))
        for name, parameter in list(child._parameters.items()):
            if parameter is not None:
                child._parameters[name] = nn.Parameter(parameter.detach().clone(), requires_grad=False)


def autograd_safe_tree(value: Any) -> Any:
    """Clone cached inference-mode inputs into ordinary tensors for replay.

    The calibration cache is intentionally read under inference mode.  Those
    tensors are fine for evaluation but cannot be retained by autograd during
    the coupled attention backward pass (including by a frozen low-rank A
    projection).  This is an in-memory replay conversion only.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: autograd_safe_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(autograd_safe_tree(item) for item in value)
    if isinstance(value, list):
        return [autograd_safe_tree(item) for item in value]
    return value


def parent_smoother(attn: torch.nn.Module):
    matches = []
    for hook in attn._forward_pre_hooks.values():
        processor = getattr(hook, "processor", None)
        if type(processor).__name__ == "ActivationSmoother":
            matches.append(processor)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one parent ActivationSmoother, found {len(matches)}")
    return matches[0]


def projection_outputs(attn: torch.nn.Module, kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = kwargs["hidden_states"]
    q_input = parent_smoother(attn).process(hidden)
    query = attn.to_q(q_input)
    if "encoder_hidden_states" in kwargs:
        key = attn.to_k(kwargs["encoder_hidden_states"])
    else:
        key = attn.to_k(q_input)
    return query, key


@torch.no_grad()
def evaluate(attn: torch.nn.Module, rows: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    sums = {name: [0.0, 0.0] for name in ("q", "k", "attn")}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for row in rows:
            kwargs = to_device(row["kwargs"], device)
            q, k = projection_outputs(attn, kwargs)
            output = tensor_output(attn(**kwargs))
            pooled_update(sums["q"], q, row["q_ref"].to(device))
            pooled_update(sums["k"], k, row["k_ref"].to(device))
            pooled_update(sums["attn"], output, row["attn_ref"].to(device))
            del kwargs, q, k, output
    return {f"{name}_nmse": value[0] / max(value[1], 1e-30) for name, value in sums.items()}


def train_pair(
    attn: torch.nn.Module,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    mode: str,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, float]]:
    materialize_frozen_params_for_backward(attn)
    q_branch, k_branch = low_rank_branch(attn.to_q), low_rank_branch(attn.to_k)
    # PTQ loading is inference-only, so its Parameters are inference tensors
    # and cannot be updated by an optimizer. Replace only the two trainable B
    # matrices with ordinary fp32 Parameters; branch topology stays unchanged.
    q_branch.b.weight = nn.Parameter(q_branch.b.weight.detach().float().clone(), requires_grad=True)
    k_branch.b.weight = nn.Parameter(k_branch.b.weight.detach().float().clone(), requires_grad=True)
    params = [q_branch.b.weight, k_branch.b.weight]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    history = []
    best_metric = float("inf")
    best = None

    for epoch in range(epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        losses = []
        for index in order:
            row = train_rows[index]
            # Cached kwargs originate in torch.inference_mode().  Coupled
            # training needs a normal tensor graph even though the inputs are
            # constants, because frozen layers save them for B-matrix grads.
            kwargs = autograd_safe_tree(to_device(row["kwargs"], device))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if mode == "independent":
                    q, k = projection_outputs(attn, kwargs)
                    loss = nmse(q, row["q_ref"].to(device)) + nmse(k, row["k_ref"].to(device))
                elif mode == "coupled":
                    output = tensor_output(attn(**kwargs))
                    loss = nmse(output, row["attn_ref"].to(device))
                else:
                    raise ValueError(mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            del kwargs, loss
        metrics = evaluate(attn, val_rows, device)
        record = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **metrics}
        history.append(record)
        print(f"[train:{mode}] {record}", flush=True)
        if metrics["attn_nmse"] < best_metric:
            best_metric = metrics["attn_nmse"]
            best = {
                "q_b": q_branch.b.weight.detach().cpu().clone(),
                "k_b": k_branch.b.weight.detach().cpu().clone(),
            }
    assert best is not None
    q_branch.b.weight.data.copy_(best["q_b"].to(device=device, dtype=q_branch.b.weight.dtype))
    k_branch.b.weight.data.copy_(best["k_b"].to(device=device, dtype=k_branch.b.weight.dtype))
    final_metrics = evaluate(attn, val_rows, device)
    return best, history, final_metrics


def put_pair_into_branch_state(
    state: dict[str, dict[str, torch.Tensor]], pair_name: str, best: dict[str, torch.Tensor]
) -> None:
    block, attn_name = pair_name.rsplit(".", 1)
    hidden = best["q_b"].shape[0]
    if attn_name == "attn1":
        key = f"{pair_name}.to_q"
        packed = state[key]["b.weight"]
        if packed.shape[0] != hidden * 3:
            raise RuntimeError(f"{key}: unexpected packed B shape {tuple(packed.shape)}")
        packed[:hidden].copy_(best["q_b"].to(packed.dtype))
        packed[hidden : 2 * hidden].copy_(best["k_b"].to(packed.dtype))
    elif attn_name == "attn2":
        q_key, kv_key = f"{pair_name}.to_q", f"{pair_name}.to_k"
        state[q_key]["b.weight"].copy_(best["q_b"].to(state[q_key]["b.weight"].dtype))
        packed = state[kv_key]["b.weight"]
        if packed.shape[0] != hidden * 2:
            raise RuntimeError(f"{kv_key}: unexpected packed B shape {tuple(packed.shape)}")
        packed[:hidden].copy_(best["k_b"].to(packed.dtype))
    else:
        raise ValueError(pair_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-prompts", type=int, default=2)
    parser.add_argument("--val-prompts", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.train_prompts <= 0 or args.val_prompts <= 0:
        raise ValueError("train-prompts and val-prompts must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    total_prompts = args.train_prompts + args.val_prompts
    paths = cache_paths(total_prompts)
    records = collect_bf16_records(paths, PAIR_NAMES, device)
    train_count = args.train_prompts * 4

    print("[quant] loading baseline checkpoint", flush=True)
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    load_helper().load_quantized_transformer(pipe, CKPT, MODEL)
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    for parameter in pipe.transformer.parameters():
        parameter.requires_grad_(False)
    modules = dict(pipe.transformer.named_modules())

    source_branch = torch.load(CKPT / "branch.pt", map_location="cpu", weights_only=False)
    output_states = {
        "independent": copy.deepcopy(source_branch),
        "coupled": copy.deepcopy(source_branch),
    }
    report: dict[str, Any] = {
        "definition": (
            "B-only fixed-budget refinement from the original SVDQuant checkpoint. "
            "Independent minimizes Q/K projection NMSE; coupled differentiates full attention OutputsError "
            "jointly into Q/K. All validation metrics use held-out original rCM calibration prompts."
        ),
        "base_checkpoint": str(CKPT),
        "cache_dir": str(CACHE_DIR),
        "pairs": list(PAIR_NAMES),
        "train_cache_files": [path.name for path in paths[:train_count]],
        "val_cache_files": [path.name for path in paths[train_count:]],
        "epochs": args.epochs,
        "lr": args.lr,
        "parameter_budget_changed": False,
        "results": {},
    }

    for pair_index, pair_name in enumerate(PAIR_NAMES):
        print(f"[pair] {pair_name}", flush=True)
        attn = modules[pair_name]
        train_rows = records[pair_name][:train_count]
        val_rows = records[pair_name][train_count:]
        q_branch, k_branch = low_rank_branch(attn.to_q), low_rank_branch(attn.to_k)
        initial = {
            "q_b": q_branch.b.weight.detach().cpu().clone(),
            "k_b": k_branch.b.weight.detach().cpu().clone(),
        }
        baseline = evaluate(attn, val_rows, device)
        pair_report = {"baseline": baseline}
        print(f"[baseline] {pair_name} {baseline}", flush=True)

        for mode in ("independent", "coupled"):
            q_branch.b.to(dtype=torch.bfloat16)
            k_branch.b.to(dtype=torch.bfloat16)
            q_branch.b.weight.data.copy_(initial["q_b"].to(device=device, dtype=torch.bfloat16))
            k_branch.b.weight.data.copy_(initial["k_b"].to(device=device, dtype=torch.bfloat16))
            best, history, metrics = train_pair(
                attn,
                train_rows,
                val_rows,
                mode,
                args.epochs,
                args.lr,
                args.seed + pair_index,
                device,
            )
            put_pair_into_branch_state(output_states[mode], pair_name, best)
            pair_report[mode] = {"history": history, "best_val": metrics}
        report["results"][pair_name] = pair_report

    for mode, state in output_states.items():
        torch.save(state, output_dir / f"branch_{mode}.pt")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved quick Q/K refinement experiment -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
