#!/usr/bin/env python3
"""Jointly refine all existing rCM-Wan LowRank branches with a denoiser loss.

Unlike the block-wise pilot, one student forward traverses all 30 quantized
blocks.  The objective is the single-timestep BF16 denoiser-output NMSE on
identical cached inputs.  W4A4 weights, smoothing, and LowRank rank are fixed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from diffusers import WanPipeline

from exp_rcm_blockwise_lowrank_pilot import (
    EPS, TEST_PROMPTS, TRAIN_PROMPTS, all_branches, freeze_all,
    install_activation_ste, load_items, restore_activation_processors,
)
from exp_rcm_signed_alignment_audit import make_autograd_safe
from exp_rcm_trajectory_sensitivity_audit import CACHE, CKPT, MODEL, as_tensor, json_default
from infer_rcm_wan_4step import load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/rcm_fullmodel_lowrank_denoiser"


def tree_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(tree_device(item, device) for item in value)
    if isinstance(value, list):
        return [tree_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: tree_device(item, device) for key, item in value.items()}
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def capture_teacher(
    teacher: torch.nn.Module,
    items: list[tuple[str, int, dict[str, Any]]],
    device: torch.device,
) -> list[dict[str, Any]]:
    records = []
    for index, (prompt_id, step, payload) in enumerate(items):
        kwargs = payload["input_kwargs"]
        output = as_tensor(teacher(
            hidden_states=payload["input_args"][0].to(device=device, dtype=torch.bfloat16),
            timestep=kwargs["timestep"].to(device=device, dtype=torch.bfloat16),
            encoder_hidden_states=kwargs["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16),
            return_dict=False,
        ))
        records.append({
            "prompt_id": prompt_id,
            "step": step,
            "hidden": payload["input_args"][0].detach().cpu(),
            "timestep": kwargs["timestep"].detach().cpu(),
            "encoder_hidden_states": kwargs["encoder_hidden_states"].detach().cpu(),
            "target": output.detach().cpu(),
        })
        print(f"teacher {index + 1}/{len(items)} prompt={prompt_id} step={step}", flush=True)
    return records


def student_forward(model: torch.nn.Module, record: dict[str, Any], device: torch.device) -> torch.Tensor:
    return as_tensor(model(
        hidden_states=record["hidden"].to(device=device, dtype=torch.bfloat16),
        timestep=record["timestep"].to(device=device, dtype=torch.bfloat16),
        encoder_hidden_states=record["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16),
        return_dict=False,
    ))


@torch.no_grad()
def evaluate(model: torch.nn.Module, records: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    total_err2 = total_ref2 = 0.0
    total_numel = 0
    rows = []
    for record in records:
        output = student_forward(model, record, device).float()
        target = record["target"].to(device=device).float()
        error = output - target
        err2, ref2 = float(error.square().sum()), float(target.square().sum())
        rows.append({"prompt_id": record["prompt_id"], "step": record["step"],
                     "nmse": err2 / max(ref2, EPS), "mse": err2 / target.numel(),
                     "err2": err2, "ref2": ref2, "numel": target.numel()})
        total_err2 += err2; total_ref2 += ref2; total_numel += target.numel()
    return {"nmse": total_err2 / max(total_ref2, EPS), "mse": total_err2 / total_numel,
            "err2": total_err2, "ref2": total_ref2, "numel": total_numel}, rows


def branch_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], dict[int, list[str]]]:
    params: dict[int, torch.nn.Parameter] = {}
    owners: dict[int, list[str]] = defaultdict(list)
    for block_index, block in enumerate(model.blocks):
        for name, module in block.named_modules():
            for hook in list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values()):
                branch = getattr(hook, "branch", None)
                if type(branch).__name__ != "LowRankBranch":
                    continue
                for parameter in branch.parameters():
                    parameter.requires_grad_(True)
                    params[id(parameter)] = parameter
                    owners[id(parameter)].append(f"blocks.{block_index}.{name}")
    if len(owners) != 510:
        raise RuntimeError(f"expected 510 unique LowRank parameter tensors, found {len(owners)}")
    return list(params.values()), owners


def branch_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for block_index, block in enumerate(model.blocks):
        for name, module in block.named_modules():
            for hook in list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values()):
                branch = getattr(hook, "branch", None)
                if type(branch).__name__ == "LowRankBranch":
                    for key, value in branch.state_dict().items():
                        result[f"blocks.{block_index}.{name}.{key}"] = value.detach().float().cpu().clone()
    if len(result) != 600:
        raise RuntimeError(f"expected 600 LowRank state tensors, found {len(result)}")
    return result


def export_by_block(model: torch.nn.Module) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    from exp_rcm_blockwise_lowrank_pilot import branch_map
    return {
        index: {name: {key: value.detach().cpu().to(torch.bfloat16) for key, value in branch.state_dict().items()}
                for name, branch in branch_map(block).items()}
        for index, block in enumerate(model.blocks)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--step-weights", type=float, nargs=4, default=[1, 1, 1, 1])
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--extra-train-cache-dir", type=Path, required=True)
    parser.add_argument("--extra-train-prompts", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()
    if args.epochs <= 0 or args.lr <= 0 or len(args.step_weights) != 4 or any(x <= 0 for x in args.step_weights):
        raise ValueError("epochs/lr/step-weights are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    train_items = load_items(args.cache_dir, TRAIN_PROMPTS) + load_items(args.extra_train_cache_dir, tuple(args.extra_train_prompts))
    test_items = load_items(args.cache_dir, TEST_PROMPTS)
    if args.max_train_samples: train_items = train_items[:args.max_train_samples]
    if args.max_test_samples: test_items = test_items[:args.max_test_samples]
    if len(train_items) != 32 and not args.max_train_samples:
        raise RuntimeError(f"expected 32 training records, got {len(train_items)}")

    teacher_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    teacher_pipe.text_encoder.to("cpu"); teacher_pipe.vae.to("cpu")
    freeze_all(teacher_pipe.transformer)
    records = capture_teacher(teacher_pipe.transformer, train_items + test_items, device)
    teacher_pipe.to("cpu"); del teacher_pipe; gc.collect(); torch.cuda.empty_cache()
    train_records, test_records = records[:len(train_items)], records[len(train_items):]

    student_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    student_pipe.text_encoder.to("cpu"); student_pipe.vae.to("cpu")
    load_quantized_transformer(student_pipe, args.checkpoint, MODEL)
    student = student_pipe.transformer
    freeze_all(student)
    student.enable_gradient_checkpointing()
    safe_counts = make_autograd_safe(student)
    params, owners = branch_parameters(student)
    before_state = branch_state(student)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    before_train, before_train_rows = evaluate(student, train_records, device)
    before_test, before_test_rows = evaluate(student, test_records, device)
    originals, ste_count = install_activation_ste(student)
    ste_value, _ = evaluate(student, train_records[:1], device)
    restore_activation_processors(originals)
    raw_value, _ = evaluate(student, train_records[:1], device)
    if ste_value["err2"] != raw_value["err2"]:
        raise RuntimeError("activation STE changed forward output")
    originals, _ = install_activation_ste(student)

    curve: list[dict[str, Any]] = []
    first_grad: dict[int, float] = defaultdict(float)
    for epoch in range(args.epochs):
        order = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_nmse, epoch_objective = [], []
        for order_index, record_index in enumerate(order):
            record = train_records[record_index]
            optimizer.zero_grad(set_to_none=True)
            output = student_forward(student, record, device).float()
            target = record["target"].to(device=device).float()
            nmse = (output - target).square().sum() / target.square().sum().clamp_min(EPS)
            objective = nmse * args.step_weights[record["step"]]
            objective.backward()
            if epoch == 0 and order_index == 0:
                for parameter in params:
                    if parameter.grad is not None:
                        first_grad[id(parameter)] += float(parameter.grad.float().square().sum())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_nmse.append(float(nmse.detach())); epoch_objective.append(float(objective.detach()))
            del output, target, nmse, objective
        curve.append({"epoch": epoch + 1, "mean_train_nmse": sum(epoch_nmse) / len(epoch_nmse),
                      "mean_weighted_objective": sum(epoch_objective) / len(epoch_objective)})
        print(json.dumps(curve[-1]), flush=True)

    missing = [name for pid, names in owners.items() for name in names if first_grad[pid] <= 0]
    if missing:
        raise RuntimeError(f"LowRank branches without first-step gradient: {sorted(set(missing))}")
    restore_activation_processors(originals)
    after_train, after_train_rows = evaluate(student, train_records, device)
    after_test, after_test_rows = evaluate(student, test_records, device)
    after_state = branch_state(student)
    unchanged = [key for key in before_state if torch.equal(before_state[key], after_state[key])]
    if len(unchanged) == len(before_state):
        raise RuntimeError("optimizer did not change any LowRank parameter")

    audit = {
        "objective": "single-timestep full quantized-transformer denoiser-output NMSE to BF16 teacher",
        "train_prompts": list(TRAIN_PROMPTS) + list(args.extra_train_prompts), "test_prompts": list(TEST_PROMPTS),
        "train_records": len(train_records), "test_records": len(test_records), "epochs": args.epochs,
        "activation_qdq": "exact forward, STE backward", "ste_forward_err2_abs_diff": abs(ste_value["err2"] - raw_value["err2"]),
        "autograd_safe_clones": safe_counts, "lowrank_parameter_tensors": len(params),
        "lowrank_state_tensors_changed": len(before_state) - len(unchanged),
        "first_step_nonzero_gradient_tensors": sum(first_grad[pid] > 0 for pid in owners),
        "first_step_nonzero_gradient_branches": len(set(name for pid, names in owners.items() if first_grad[pid] > 0 for name in names)),
        "train_before": before_train, "train_after": after_train, "test_before": before_test, "test_after": after_test,
    }
    (args.output_dir / "config.json").write_text(json.dumps(vars(args) | audit, indent=2, default=json_default))
    write_csv(args.output_dir / "training_curve.csv", curve)
    rows = []
    for split, phase, values in (("train", "before", before_train_rows), ("train", "after", after_train_rows),
                                 ("test", "before", before_test_rows), ("test", "after", after_test_rows)):
        rows.extend({"split": split, "phase": phase, **value} for value in values)
    write_csv(args.output_dir / "per_sample_nmse.csv", rows)
    torch.save(export_by_block(student), args.output_dir / "trained_lowrank_branches.pt")
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in curve], [row["mean_train_nmse"] for row in curve], marker="o")
    axis.set(xlabel="epoch", ylabel="online denoiser NMSE", yscale="log")
    axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(args.output_dir / "loss_curve.png", dpi=180); plt.close(fig)
    print(json.dumps(audit, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
