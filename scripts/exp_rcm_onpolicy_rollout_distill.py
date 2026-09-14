#!/usr/bin/env python3
"""On-policy single-step rollout distillation for rCM-Wan LowRank branches.

For each microbatch, the W4A4 model first rolls out without gradients to a
randomly selected rCM timestep.  BF16 and W4A4 then see the same *quantized*
latent state; only the W4A4 current-step DiT output is differentiated.  Thus
there is no four-step backward graph and no saved rollout tensors.
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
    EPS, TEST_PROMPTS, TRAIN_PROMPTS, freeze_all, install_activation_ste,
    load_items, restore_activation_processors,
)
from exp_rcm_fullmodel_lowrank_denoiser import (
    branch_parameters, branch_state, export_by_block,
)
from exp_rcm_signed_alignment_audit import make_autograd_safe
from exp_rcm_trajectory_sensitivity_audit import CACHE, CKPT, MODEL, as_tensor, json_default
from infer_rcm_wan_4step import load_quantized_transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/rcm_onpolicy_rollout_distill"


def rcm_times(device: torch.device) -> torch.Tensor:
    values = torch.tensor([torch.atan(torch.tensor(80.0)), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    return torch.sin(values) / (torch.cos(values) + torch.sin(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def conditions(cache_dir: Path, prompt_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for prompt_id, step, payload in load_items(cache_dir, prompt_ids):
        if prompt_id in result:
            continue
        kwargs = payload["input_kwargs"]
        result[prompt_id] = {
            "encoder_hidden_states": kwargs["encoder_hidden_states"].detach().cpu(),
            "shape": tuple(payload["input_args"][0].shape),
        }
    if set(result) != set(prompt_ids):
        raise RuntimeError(f"condition mismatch: got={sorted(result)} expected={sorted(prompt_ids)}")
    return result


def model_forward(model: torch.nn.Module, latent: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    return as_tensor(model(
        hidden_states=latent.to(dtype=torch.bfloat16),
        timestep=timestep.to(dtype=torch.bfloat16),
        encoder_hidden_states=cond.to(dtype=torch.bfloat16),
        return_dict=False,
    ))


def initial_and_noises(shape: tuple[int, ...], seed: int, times: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(torch.float64) * times[0]
    noises = [torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(torch.float64) for _ in range(4)]
    return initial, noises


@torch.no_grad()
def quant_prefix(student: torch.nn.Module, latent: torch.Tensor, noises: list[torch.Tensor], times: torch.Tensor,
                 cond: torch.Tensor, target_step: int) -> torch.Tensor:
    one = torch.ones((1,), device=latent.device, dtype=torch.float64)
    state = latent
    for step in range(target_step):
        current, nxt = times[step], times[step + 1]
        timestep = current.float() * one * 1000
        velocity = model_forward(student, state, timestep, cond).to(torch.float64)
        state = (1 - nxt) * (state - current * velocity) + nxt * noises[step]
    return state.detach()


@torch.no_grad()
def teacher_output(teacher: torch.nn.Module, state: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    timestep = time.float() * torch.ones((1,), device=state.device, dtype=torch.float64) * 1000
    return model_forward(teacher, state, timestep, cond).detach()


@torch.no_grad()
def evaluate_rollout(student: torch.nn.Module, teacher: torch.nn.Module, conds: dict[str, dict[str, Any]],
                     seeds: list[int], device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    times = rcm_times(device)
    rows: list[dict[str, Any]] = []
    for prompt_id, record in conds.items():
        cond = record["encoder_hidden_states"].to(device=device)
        for seed in seeds:
            bf_state, noises = initial_and_noises(record["shape"], seed, times, device)
            q_state = bf_state.clone()
            for step in range(4):
                current, nxt = times[step], times[step + 1]
                timestep = current.float() * torch.ones((1,), device=device, dtype=torch.float64) * 1000
                bf_v = model_forward(teacher, bf_state, timestep, cond).to(torch.float64)
                q_v = model_forward(student, q_state, timestep, cond).to(torch.float64)
                v_err2, v_ref2 = float((q_v - bf_v).square().sum()), float(bf_v.square().sum())
                bf_state = (1 - nxt) * (bf_state - current * bf_v) + nxt * noises[step]
                q_state = (1 - nxt) * (q_state - current * q_v) + nxt * noises[step]
                x_err2, x_ref2 = float((q_state - bf_state).square().sum()), float(bf_state.square().sum())
                rows.append({"prompt_id": prompt_id, "seed": seed, "step": step,
                             "rollout_v_nmse": v_err2 / max(v_ref2, EPS),
                             "latent_nmse": x_err2 / max(x_ref2, EPS),
                             "v_err2": v_err2, "v_ref2": v_ref2, "latent_err2": x_err2, "latent_ref2": x_ref2})
    final = [row for row in rows if row["step"] == 3]
    summary = {
        "rollout_v_nmse": sum(row["v_err2"] for row in rows) / max(sum(row["v_ref2"] for row in rows), EPS),
        "final_latent_nmse": sum(row["latent_err2"] for row in final) / max(sum(row["latent_ref2"] for row in final), EPS),
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--extra-train-cache-dir", type=Path, required=True)
    ap.add_argument("--extra-train-prompts", nargs="+", required=True)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[9101])
    args = ap.parse_args()
    if args.updates <= 0 or args.accumulation <= 0 or args.lr <= 0:
        raise ValueError("updates/accumulation/lr must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda")
    train_ids = tuple(TRAIN_PROMPTS) + tuple(args.extra_train_prompts)
    train_conds = conditions(args.cache_dir, tuple(TRAIN_PROMPTS))
    train_conds.update(conditions(args.extra_train_cache_dir, tuple(args.extra_train_prompts)))
    test_conds = conditions(args.cache_dir, tuple(TEST_PROMPTS))

    teacher_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    teacher_pipe.text_encoder.to("cpu"); teacher_pipe.vae.to("cpu")
    teacher = teacher_pipe.transformer; freeze_all(teacher)

    student_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    student_pipe.text_encoder.to("cpu"); student_pipe.vae.to("cpu")
    load_quantized_transformer(student_pipe, args.checkpoint, MODEL)
    student = student_pipe.transformer; freeze_all(student); student.enable_gradient_checkpointing()
    safe_counts = make_autograd_safe(student)
    params, owners = branch_parameters(student)
    before_state = branch_state(student)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    originals, ste_count = install_activation_ste(student)

    before_train, before_train_rows = evaluate_rollout(student, teacher, train_conds, args.eval_seeds, device)
    before_test, before_test_rows = evaluate_rollout(student, teacher, test_conds, args.eval_seeds, device)
    print(json.dumps({"before_train": before_train, "before_test": before_test}), flush=True)
    times = rcm_times(device)
    rng = random.Random(args.seed)
    curve: list[dict[str, Any]] = []
    first_grad: dict[int, float] = defaultdict(float)
    micro_index = 0
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        selected_steps: list[int] = []
        for accum_index in range(args.accumulation):
            prompt_id = rng.choice(train_ids)
            target_step = rng.randrange(4)
            rollout_seed = args.seed * 1_000_000 + micro_index
            micro_index += 1
            record = train_conds[prompt_id]
            cond = record["encoder_hidden_states"].to(device=device)
            initial, noises = initial_and_noises(record["shape"], rollout_seed, times, device)
            state = quant_prefix(student, initial, noises, times, cond, target_step)
            target = teacher_output(teacher, state, times[target_step], cond).float()
            timestep = times[target_step].float() * torch.ones((1,), device=device, dtype=torch.float64) * 1000
            output = model_forward(student, state, timestep, cond).float()
            nmse = (output - target).square().sum() / target.square().sum().clamp_min(EPS)
            (nmse / args.accumulation).backward()
            micro_losses.append(float(nmse.detach())); selected_steps.append(target_step)
            del initial, noises, state, target, output, nmse
        if update == 1:
            for parameter in params:
                if parameter.grad is not None:
                    first_grad[id(parameter)] += float(parameter.grad.float().square().sum())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        row = {"update": update, "mean_onpolicy_nmse": sum(micro_losses) / len(micro_losses),
               "grad_norm_preclip": grad_norm, "steps": ",".join(map(str, selected_steps))}
        curve.append(row)
        print(json.dumps(row), flush=True)
        if update % args.eval_every == 0 or update == args.updates:
            train_summary, _ = evaluate_rollout(student, teacher, train_conds, args.eval_seeds, device)
            test_summary, _ = evaluate_rollout(student, teacher, test_conds, args.eval_seeds, device)
            row.update({f"train_{key}": value for key, value in train_summary.items()})
            row.update({f"test_{key}": value for key, value in test_summary.items()})
            print(json.dumps({"update": update, "eval_train": train_summary, "eval_test": test_summary}), flush=True)
            torch.save(export_by_block(student), args.output_dir / f"checkpoint_update{update:03d}.pt")

    missing = [name for pid, names in owners.items() for name in names if first_grad[pid] <= 0]
    if missing:
        raise RuntimeError(f"LowRank branches without first-update gradient: {sorted(set(missing))}")
    after_train, after_train_rows = evaluate_rollout(student, teacher, train_conds, args.eval_seeds, device)
    after_test, after_test_rows = evaluate_rollout(student, teacher, test_conds, args.eval_seeds, device)
    restore_activation_processors(originals)
    after_state = branch_state(student)
    changed = sum(not torch.equal(before_state[key], after_state[key]) for key in before_state)
    if not changed:
        raise RuntimeError("optimizer did not change any LowRank state tensor")
    audit = {
        "objective": "on-policy single-step DiT-output NMSE: BF16 and W4A4 receive the same W4A4 rollout latent",
        "prefix": "no_grad W4A4 rollout with fresh seed per microbatch; no rollout tensors saved",
        "train_prompts": list(train_ids), "test_prompts": list(TEST_PROMPTS),
        "activation_qdq": "exact forward, STE backward", "activation_quantizers": ste_count,
        "autograd_safe_clones": safe_counts, "accumulation": args.accumulation,
        "lowrank_parameter_tensors": len(params), "lowrank_state_tensors_changed": changed,
        "first_update_nonzero_gradient_tensors": sum(first_grad[pid] > 0 for pid in owners),
        "before_train": before_train, "after_train": after_train,
        "before_test": before_test, "after_test": after_test,
    }
    write_csv(args.output_dir / "training_curve.csv", curve)
    rows = []
    for split, phase, values in (("train", "before", before_train_rows), ("test", "before", before_test_rows),
                                 ("train", "after", after_train_rows), ("test", "after", after_test_rows)):
        rows.extend({"split": split, "phase": phase, **value} for value in values)
    write_csv(args.output_dir / "rollout_metrics.csv", rows)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args) | audit, indent=2, default=json_default))
    torch.save(export_by_block(student), args.output_dir / "trained_lowrank_branches.pt")
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["update"] for row in curve], [row["mean_onpolicy_nmse"] for row in curve], marker="o", ms=3)
    axis.set(xlabel="optimizer update", ylabel="on-policy DiT-output NMSE", yscale="log")
    axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.output_dir / "loss_curve.png", dpi=180); plt.close(fig)
    print(json.dumps(audit, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
