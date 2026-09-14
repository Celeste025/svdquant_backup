#!/usr/bin/env python3
"""Evaluate the original rCM SVDQuant checkpoint separately on train/held-out caches."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel

from exp_rcm_blockwise_lowrank_pilot import TEST_PROMPTS, TRAIN_PROMPTS, load_items
from exp_rcm_fullmodel_lowrank_denoiser import as_tensor
from exp_rcm_joint_smooth_lowrank import CACHE, CKPT, EXTRA_CACHE, EXTRA_PROMPTS, MODEL, RCM_TRANSFORMER
from infer_rcm_wan_4step import load_quantized_transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/rcm_joint_smooth_lowrank/full_20260908/baseline_split"


def run(model: torch.nn.Module, split: str, items, device: torch.device):
    rows = []; totals = [0.0, 0.0, 0]
    with torch.inference_mode():
        for index, (prompt, step, payload) in enumerate(items):
            kw = payload["input_kwargs"]
            got = as_tensor(model(hidden_states=payload["input_args"][0].to(device=device, dtype=torch.bfloat16),
                                  timestep=kw["timestep"].to(device=device, dtype=torch.bfloat16),
                                  encoder_hidden_states=kw["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16),
                                  return_dict=False)).float()
            ref = as_tensor(payload["outputs"]).to(device=device).float()
            err2, ref2 = float((got-ref).square().sum()), float(ref.square().sum())
            rows.append({"split": split, "prompt_id": prompt, "step": step, "nmse": err2/ref2,
                         "mse": err2/ref.numel(), "err2": err2, "ref2": ref2, "numel": ref.numel()})
            totals[0] += err2; totals[1] += ref2; totals[2] += ref.numel()
            print(f"{split} {index+1}/{len(items)} {prompt} step={step}", flush=True)
    return rows, {"nmse": totals[0]/totals[1], "mse": totals[0]/totals[2], "err2": totals[0], "ref2": totals[1], "numel": totals[2]}


def main():
    OUT.mkdir(parents=True, exist_ok=True); device=torch.device("cuda")
    train = load_items(CACHE, TRAIN_PROMPTS) + load_items(EXTRA_CACHE, EXTRA_PROMPTS)
    test = load_items(CACHE, TEST_PROMPTS)
    pipe=WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.transformer=WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    pipe.to(device); pipe.text_encoder.to("cpu"); pipe.vae.to("cpu")
    load_quantized_transformer(pipe, CKPT, MODEL)
    rows_a, sum_a=run(pipe.transformer,"train",train,device)
    rows_b, sum_b=run(pipe.transformer,"heldout",test,device)
    with (OUT/"per_sample_baseline_nmse.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_a[0])); w.writeheader(); w.writerows(rows_a+rows_b)
    (OUT/"baseline_split_summary.json").write_text(json.dumps({"train":sum_a,"heldout":sum_b},indent=2))
    print(json.dumps({"train":sum_a,"heldout":sum_b},indent=2))

if __name__ == "__main__": main()
