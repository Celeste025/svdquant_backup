#!/usr/bin/env python3
"""Audit rollout-v error scale across train and held-out rCM cache prompts.

This deliberately records raw per-element MSE and the BF16 output energy in
addition to NMSE; a changing NMSE denominator cannot be mistaken for a change
in absolute quantization error.
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel

from exp_rcm_blockwise_lowrank_pilot import TRAIN_PROMPTS, TEST_PROMPTS, freeze_all, install_activation_ste, restore_activation_processors
from exp_rcm_joint_smooth_lowrank import CACHE, CKPT, MODEL, RCM_TRANSFORMER, DynamicSmooth
from exp_rcm_onpolicy_rollout_distill import conditions, initial_and_noises, model_forward, rcm_times
from infer_rcm_joint_full20 import load_joint_state, load_quantized_transformer

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/reports/rcm_joint_smooth_lowrank/full20_20260909/epoch_10.pt"
OUT = ROOT / "results/reports/rcm_joint_full20_unseen/epoch10_rollout_multicase"
SEED = 9101

def stats(q: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    err = float((q.float() - ref.float()).square().mean().item())
    energy = float(ref.float().square().mean().item())
    return err, energy, err / max(energy, 1e-12)

def main() -> None:
    device = torch.device("cuda")
    train = conditions(CACHE, TRAIN_PROMPTS)
    heldout = conditions(CACHE, TEST_PROMPTS)
    records = [("train", prompt, train[prompt]) for prompt in TRAIN_PROMPTS]
    records += [("heldout", prompt, heldout[prompt]) for prompt in TEST_PROMPTS]

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    pipe.to(device)
    load_quantized_transformer(pipe, CKPT, MODEL)
    student = pipe.transformer
    freeze_all(student)
    teacher = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16).to(device)
    freeze_all(teacher)
    dynamic = DynamicSmooth(student, teacher, CKPT, bound=1.25).to(device)
    originals, _ = install_activation_ste(student)
    load_joint_state(student, dynamic, STATE)
    pipe.text_encoder.to("cpu"); pipe.vae.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    times = rcm_times(device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for case_index, (split, prompt_id, record) in enumerate(records, 1):
            cond = record["encoder_hidden_states"].to(device=device)
            bf_state, noises = initial_and_noises(record["shape"], SEED, times, device)
            q_state = bf_state.clone()
            for step, (current, nxt) in enumerate(zip(times[:-1], times[1:])):
                timestep = current.float() * torch.ones((1,), device=device, dtype=torch.float64) * 1000
                bf_v = model_forward(teacher, bf_state, timestep, cond).to(torch.float64)
                q_v = model_forward(student, q_state, timestep, cond).to(torch.float64)
                v_mse, v_energy, v_nmse = stats(q_v, bf_v)
                bf_next = (1 - nxt) * (bf_state - current * bf_v) + nxt * noises[step]
                q_next = (1 - nxt) * (q_state - current * q_v) + nxt * noises[step]
                x_mse, x_energy, x_nmse = stats(q_next, bf_next)
                rows.append({"split": split, "prompt_id": prompt_id, "seed": SEED, "step": step,
                    "t_current": float(current), "rollout_v_raw_mse": v_mse,
                    "bf16_v_energy": v_energy, "rollout_v_nmse": v_nmse,
                    "updated_latent_raw_mse": x_mse, "bf16_latent_energy": x_energy,
                    "updated_latent_nmse": x_nmse})
                bf_state, q_state = bf_next, q_next
            print(f"completed {case_index}/{len(records)} {split}/{prompt_id}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for split in ("train", "heldout", "all"):
        selected = rows if split == "all" else [r for r in rows if r["split"] == split]
        for step in range(4):
            items = [r for r in selected if r["step"] == step]
            aggregate.append({"split": split, "step": step, "cases": len(items),
                "mean_rollout_v_raw_mse": sum(float(r["rollout_v_raw_mse"]) for r in items) / len(items),
                "mean_bf16_v_energy": sum(float(r["bf16_v_energy"]) for r in items) / len(items),
                "mean_rollout_v_nmse": sum(float(r["rollout_v_nmse"]) for r in items) / len(items)})
    with OUT.with_name(OUT.name + "_aggregate").with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    OUT.with_suffix(".json").write_text(json.dumps({"checkpoint": str(STATE), "seed": SEED,
        "train_prompts": list(TRAIN_PROMPTS), "heldout_prompts": list(TEST_PROMPTS),
        "definition": "q_v(q_rollout_latent) vs bf16_v(q_rollout_latent); values are per-element means",
        "rows": rows, "aggregate": aggregate}, indent=2) + "\n")
    restore_activation_processors(originals)
    print("saved", OUT)

if __name__ == "__main__":
    main()
