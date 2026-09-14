#!/usr/bin/env python3
"""Train existing SVDQuant LowRank branches jointly at Wan block endpoints."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from diffusers import WanPipeline

from exp_rcm_signed_alignment_audit import make_autograd_safe
from exp_rcm_trajectory_sensitivity_audit import CACHE, CKPT, EPS, MODEL, as_tensor, json_default
from infer_rcm_wan_4step import load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/rcm_blockwise_lowrank_pilot"
TRAIN_PROMPTS = ("0015", "0005", "0006", "0003")
TEST_PROMPTS = ("0001", "0002", "0007", "0009")


@dataclass
class Record:
    prompt_id: str
    step: int
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    target: torch.Tensor


def tree_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, tuple):
        return tuple(tree_cpu(item) for item in value)
    if isinstance(value, list):
        return [tree_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    return value


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


def load_items(cache_dir: Path, prompt_ids: tuple[str, ...]) -> list[tuple[str, int, dict[str, Any]]]:
    result = []
    for prompt_id in prompt_ids:
        paths = sorted(cache_dir.glob(f"{prompt_id}-*.pt"))
        items = []
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            items.append((int(payload["step"]), payload))
        items.sort(key=lambda item: item[0])
        if [step for step, _ in items] != [0, 1, 2, 3]:
            raise RuntimeError(f"{prompt_id}: expected steps 0..3, got {[step for step, _ in items]}")
        result.extend((prompt_id, step, payload) for step, payload in items)
    return result


def branch_map(block: torch.nn.Module) -> dict[str, torch.nn.Module]:
    result: dict[str, torch.nn.Module] = {}
    for name, module in block.named_modules():
        hooks = list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values())
        branches = [getattr(hook, "branch", None) for hook in hooks]
        branches = [branch for branch in branches if type(branch).__name__ == "LowRankBranch"]
        unique = {id(branch): branch for branch in branches}
        if len(unique) > 1:
            raise RuntimeError(f"{name}: multiple LowRankBranch objects")
        if unique:
            result[name] = next(iter(unique.values()))
    if len(result) != 10:
        raise RuntimeError(f"expected 10 LowRank branches in block, found {len(result)}: {sorted(result)}")
    return result


def all_branches(model: torch.nn.Module) -> list[torch.nn.Module]:
    found: dict[int, torch.nn.Module] = {}
    for module in model.modules():
        for hook in list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values()):
            branch = getattr(hook, "branch", None)
            if type(branch).__name__ == "LowRankBranch":
                found[id(branch)] = branch
    return list(found.values())


def freeze_all(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for branch in all_branches(model):
        for parameter in branch.parameters():
            parameter.requires_grad_(False)


def install_activation_ste(block: torch.nn.Module) -> tuple[list[Any], int]:
    """Keep exact QDQ forward values while replacing its backward by identity."""
    originals: list[Any] = []
    seen: set[int] = set()
    for module in block.modules():
        for hook in module._forward_pre_hooks.values():
            processor = getattr(hook, "processor", None)
            if "Quantizer" not in type(processor).__name__ or id(processor) in seen:
                continue
            seen.add(id(processor))
            original = processor.process

            def ste(_self: Any, tensor: torch.Tensor, _original=original) -> torch.Tensor:
                quantized = _original(tensor)
                return tensor + (quantized - tensor).detach()

            processor.process = types.MethodType(ste, processor)
            originals.append((processor, original))
    return originals, len(seen)


def restore_activation_processors(originals: list[Any]) -> None:
    for processor, original in originals:
        processor.process = original


@torch.no_grad()
def capture_records(
    quant: torch.nn.Module,
    bf: torch.nn.Module,
    items: list[tuple[str, int, dict[str, Any]]],
    block_indices: list[int],
    device: torch.device,
) -> dict[int, list[Record]]:
    records: dict[int, list[Record]] = defaultdict(list)
    current = {"prompt_id": "", "step": -1}
    handles = []

    def make_hook(index: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            target = bf.blocks[index](*args, **kwargs)
            records[index].append(Record(
                current["prompt_id"], current["step"], tree_cpu(args), tree_cpu(kwargs), target.detach().cpu()
            ))
        return hook

    for index in block_indices:
        handles.append(quant.blocks[index].register_forward_pre_hook(make_hook(index), with_kwargs=True))
    try:
        for item_index, (prompt_id, step, payload) in enumerate(items):
            current.update(prompt_id=prompt_id, step=step)
            kwargs = payload["input_kwargs"]
            quant(
                hidden_states=payload["input_args"][0].to(device=device, dtype=torch.bfloat16),
                timestep=kwargs["timestep"].to(device=device, dtype=torch.bfloat16),
                encoder_hidden_states=kwargs["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16),
                return_dict=False,
            )
            print(f"capture {item_index + 1}/{len(items)} prompt={prompt_id} step={step}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for index in block_indices:
        if len(records[index]) != len(items):
            raise RuntimeError(f"block {index}: captured {len(records[index])}/{len(items)} records")
    return records


def forward_block(block: torch.nn.Module, record: Record, device: torch.device) -> torch.Tensor:
    args = tree_device(record.args, device)
    kwargs = tree_device(record.kwargs, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return as_tensor(block(*args, **kwargs))


@torch.no_grad()
def evaluate(block: torch.nn.Module, records: list[Record], device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    total_err2 = total_ref2 = 0.0
    total_numel = 0
    rows = []
    for record in records:
        output = forward_block(block, record, device).float()
        target = record.target.to(device=device).float()
        error = output - target
        err2 = float(error.square().sum())
        ref2 = float(target.square().sum())
        total_err2 += err2
        total_ref2 += ref2
        total_numel += target.numel()
        rows.append({"prompt_id": record.prompt_id, "step": record.step, "nmse": err2 / max(ref2, EPS),
                     "mse": err2 / target.numel(), "err2": err2, "ref2": ref2, "numel": target.numel()})
        del output, target, error
    return {"nmse": total_err2 / max(total_ref2, EPS), "mse": total_err2 / total_numel,
            "err2": total_err2, "ref2": total_ref2, "numel": total_numel}, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--step-weights", type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--extra-train-cache-dir", type=Path, default=None,
                        help="Optional cache directory containing additional training prompts.")
    parser.add_argument("--extra-train-prompts", type=str, nargs="*", default=[],
                        help="Prompt IDs to load from --extra-train-cache-dir; each must contain steps 0..3.")
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()
    if len(set(args.blocks)) != len(args.blocks) or any(index < 0 or index >= 30 for index in args.blocks):
        raise ValueError(f"invalid block indices: {args.blocks}")
    if len(args.step_weights) != 4 or any(weight <= 0 for weight in args.step_weights):
        raise ValueError(f"step weights must be four positive values, got {args.step_weights}")
    if bool(args.extra_train_cache_dir) != bool(args.extra_train_prompts):
        raise ValueError("--extra-train-cache-dir and --extra-train-prompts must be supplied together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args) | {
        "train_prompts": list(TRAIN_PROMPTS) + list(args.extra_train_prompts), "test_prompts": TEST_PROMPTS,
        "base_train_cache_dir": str(args.cache_dir),
        "extra_train_cache_dir": str(args.extra_train_cache_dir) if args.extra_train_cache_dir else None,
        "samples_per_prompt": 4, "target": "BF16 block output on identical quant block input",
        "objective": "step_weights[step] * per-sample block-output NMSE",
        "trainable": "existing LowRankBranch parameters only", "activation_qdq_backward": "STE; exact QDQ forward",
    }, indent=2, default=json_default))

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    train_items = load_items(args.cache_dir, TRAIN_PROMPTS)
    if args.extra_train_cache_dir:
        train_items.extend(load_items(args.extra_train_cache_dir, tuple(args.extra_train_prompts)))
    test_items = load_items(args.cache_dir, TEST_PROMPTS)
    if args.max_train_samples:
        train_items = train_items[:args.max_train_samples]
    if args.max_test_samples:
        test_items = test_items[:args.max_test_samples]
    all_items = train_items + test_items

    pipe_bf = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    pipe_q = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    load_quantized_transformer(pipe_q, args.checkpoint, MODEL)
    bf, quant = pipe_bf.transformer, pipe_q.transformer
    freeze_all(bf)
    freeze_all(quant)
    pipe_bf.text_encoder.to("cpu"); pipe_bf.vae.to("cpu")
    pipe_q.text_encoder.to("cpu"); pipe_q.vae.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    print(f"capturing identical-input BF16 targets for blocks {args.blocks}", flush=True)
    records = capture_records(quant, bf, all_items, args.blocks, device)
    pipe_bf.transformer.to("cpu")
    del bf, pipe_bf
    safe_counts = make_autograd_safe(quant)
    freeze_all(quant)
    print(f"autograd-safe clones: {safe_counts}", flush=True)
    gc.collect(); torch.cuda.empty_cache()

    curves: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    saved_branches: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for block_index in args.blocks:
        block = quant.blocks[block_index]
        branches = branch_map(block)
        for branch in all_branches(quant):
            for parameter in branch.parameters():
                parameter.requires_grad_(False)
        parameter_by_id: dict[int, torch.nn.Parameter] = {}
        parameter_owners: dict[int, set[str]] = defaultdict(set)
        for name, branch in branches.items():
            branch.to(dtype=torch.float32)
            for parameter in branch.parameters():
                parameter.requires_grad_(True)
                parameter_by_id[id(parameter)] = parameter
                parameter_owners[id(parameter)].add(name)
        parameters = list(parameter_by_id.values())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
        block_records = records[block_index]
        train_records = block_records[:len(train_items)]
        test_records = block_records[len(train_items):]

        before_train, before_train_rows = evaluate(block, train_records, device)
        before_test, before_test_rows = evaluate(block, test_records, device)
        originals, ste_count = install_activation_ste(block)
        after_ste_train, _ = evaluate(block, train_records[:1], device)
        restore_activation_processors(originals)
        before_ste_train, _ = evaluate(block, train_records[:1], device)
        ste_forward_abs_diff = abs(after_ste_train["err2"] - before_ste_train["err2"])
        if ste_forward_abs_diff != 0:
            raise RuntimeError(f"block {block_index}: STE changed forward err2 by {ste_forward_abs_diff}")
        originals, ste_count = install_activation_ste(block)

        first_step_grad_by_branch: dict[str, float] = defaultdict(float)
        for epoch in range(args.epochs):
            order = list(range(len(train_records)))
            random.Random(args.seed + block_index * 1000 + epoch).shuffle(order)
            epoch_nmse = []
            epoch_objectives = []
            for order_index, record_index in enumerate(order):
                record = train_records[record_index]
                optimizer.zero_grad(set_to_none=True)
                output = forward_block(block, record, device).float()
                target = record.target.to(device=device).float()
                nmse = (output - target).square().sum() / target.square().sum().clamp_min(EPS)
                loss = nmse * args.step_weights[record.step]
                loss.backward()
                if epoch == 0 and order_index == 0:
                    for parameter in parameters:
                        if parameter.grad is not None:
                            energy = float(parameter.grad.float().square().sum())
                            for owner in parameter_owners[id(parameter)]:
                                first_step_grad_by_branch[owner] += energy
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                epoch_nmse.append(float(nmse.detach()))
                epoch_objectives.append(float(loss.detach()))
                del output, target, nmse, loss
            curves.append({"block": block_index, "epoch": epoch + 1,
                           "mean_train_step_nmse": sum(epoch_nmse) / len(epoch_nmse),
                           "min_train_step_nmse": min(epoch_nmse), "max_train_step_nmse": max(epoch_nmse),
                           "mean_weighted_objective": sum(epoch_objectives) / len(epoch_objectives)})
            print(f"block={block_index} epoch={epoch + 1}/{args.epochs} train_step_nmse={curves[-1]['mean_train_step_nmse']:.6g}", flush=True)

        missing_grad = sorted(name for name in branches if first_step_grad_by_branch.get(name, 0.0) <= 0)
        if missing_grad:
            raise RuntimeError(f"block {block_index}: branches without nonzero first-step gradient: {missing_grad}")
        restore_activation_processors(originals)
        after_train, after_train_rows = evaluate(block, train_records, device)
        after_test, after_test_rows = evaluate(block, test_records, device)
        for split, phase, metric_value in (
            ("train", "before", before_train), ("train", "after", after_train),
            ("test", "before", before_test), ("test", "after", after_test),
        ):
            aggregate_rows.append({"block": block_index, "split": split, "phase": phase, **metric_value})
        for split, phase, values in (
            ("train", "before", before_train_rows), ("train", "after", after_train_rows),
            ("test", "before", before_test_rows), ("test", "after", after_test_rows),
        ):
            per_sample_rows.extend({"block": block_index, "split": split, "phase": phase, **row} for row in values)
        saved_branches[block_index] = {
            name: {key: value.detach().cpu().to(torch.bfloat16) for key, value in branch.state_dict().items()}
            for name, branch in branches.items()
        }
        audit = {
            "block": block_index, "lowrank_branches": sorted(branches), "ste_activation_quantizers": ste_count,
            "unique_branch_objects": len({id(branch) for branch in branches.values()}),
            "unique_trainable_parameter_tensors": len(parameters),
            "shared_parameter_tensors": sum(len(owners) > 1 for owners in parameter_owners.values()),
            "ste_forward_err2_abs_diff": ste_forward_abs_diff,
            "first_step_grad_energy_by_branch": dict(first_step_grad_by_branch),
            "train_nmse_before": before_train["nmse"], "train_nmse_after": after_train["nmse"],
            "test_nmse_before": before_test["nmse"], "test_nmse_after": after_test["nmse"],
        }
        (args.output_dir / f"block{block_index}_audit.json").write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2), flush=True)
        del optimizer, parameters, block_records, train_records, test_records
        gc.collect(); torch.cuda.empty_cache()

    write_csv(args.output_dir / "training_curve.csv", curves)
    write_csv(args.output_dir / "aggregate_nmse.csv", aggregate_rows)
    write_csv(args.output_dir / "per_sample_nmse.csv", per_sample_rows)
    torch.save(saved_branches, args.output_dir / "trained_lowrank_branches.pt")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for block_index in args.blocks:
        values = [row for row in curves if row["block"] == block_index]
        axes[0].plot([row["epoch"] for row in values], [row["mean_train_step_nmse"] for row in values], marker="o", label=f"block {block_index}")
        for split, marker in (("train", "o"), ("test", "s")):
            before = next(row["nmse"] for row in aggregate_rows if row["block"] == block_index and row["split"] == split and row["phase"] == "before")
            after = next(row["nmse"] for row in aggregate_rows if row["block"] == block_index and row["split"] == split and row["phase"] == "after")
            x = block_index + (-.12 if split == "train" else .12)
            axes[1].plot([x, x], [before, after], color="tab:blue" if split == "train" else "tab:orange", alpha=.7)
            axes[1].scatter([x], [before], marker="x", color="black")
            axes[1].scatter([x], [after], marker=marker, color="tab:blue" if split == "train" else "tab:orange")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("mean online block NMSE"); axes[0].set_yscale("log"); axes[0].legend()
    axes[1].set_xticks(args.blocks); axes[1].set_xlabel("block"); axes[1].set_ylabel("pooled block-output NMSE")
    axes[0].grid(alpha=.2); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output_dir / "convergence_and_generalization.png", dpi=180); plt.close(fig)
    print(f"saved pilot to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
