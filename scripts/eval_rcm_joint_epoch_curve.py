#!/usr/bin/env python3
"""Evaluate selected dynamic-smoothing joint-training checkpoints on held-out cache."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel

from exp_rcm_blockwise_lowrank_pilot import TEST_PROMPTS, branch_map, freeze_all, install_activation_ste, load_items, restore_activation_processors
from exp_rcm_fullmodel_lowrank_denoiser import capture_teacher, evaluate
from exp_rcm_joint_smooth_lowrank import CACHE, CKPT, MODEL, OUT, RCM_TRANSFORMER, DynamicSmooth, load_base_anchor
from exp_rcm_signed_alignment_audit import make_autograd_safe
from infer_rcm_wan_4step import load_quantized_transformer

RUN = OUT / "full20_20260909"
EPOCHS = (5, 10, 15, 20)


def load_joint_state(student: torch.nn.Module, dynamic: DynamicSmooth, state_path: Path) -> None:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    # Match smoothing state by its shared layer-set rather than its incidental
    # construction order.  This also makes any order mismatch explicit.
    saved_groups = {tuple(entry["layers"]): delta for entry, delta in zip(
        state["groups"], state["smooth_log_delta"], strict=True
    )}
    current_groups = [tuple(entry["layers"]) for entry in dynamic.manifest()]
    if set(saved_groups) != set(current_groups):
        raise RuntimeError("saved/current smoothing group manifests differ")
    remapped = [saved_groups[group] for group in current_groups]
    print(f"smooth_group_order_equal={list(saved_groups) == current_groups}", flush=True)
    # DeepCompressor materializes original branches under inference_mode; use
    # the same mode when restoring a checkpoint to permit its in-place copies.
    with torch.inference_mode():
        for index, branches in state["lowrank"].items():
            maps = branch_map(student.blocks[int(index)])
            for name, values in branches.items():
                maps[name].load_state_dict(values)
        if len(remapped) != len(dynamic.delta):
            raise RuntimeError("smoothing group count mismatch")
        for target, source in zip(dynamic.delta, remapped, strict=True):
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    ap.add_argument("--limit-items", type=int, default=0,
                    help="for exact smoke comparison only; 0 evaluates all held-out items")
    args = ap.parse_args()
    run, epochs = args.run, tuple(args.epochs)
    # load_quantized_transformer reconstructs base LowRank branches with
    # randomized SVD because the original checkpoint has no branch.pt.
    # Match the training seed before that reconstruction.
    random.seed(20260908)
    torch.manual_seed(20260908)
    device = torch.device("cuda")
    items = load_items(CACHE, TEST_PROMPTS)
    if args.limit_items:
        items = items[:args.limit_items]
    teacher_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    teacher_pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    teacher_pipe.to(device); teacher_pipe.text_encoder.to("cpu"); teacher_pipe.vae.to("cpu")
    freeze_all(teacher_pipe.transformer)
    records = capture_teacher(teacher_pipe.transformer, items, device)

    student_pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    student_pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    student_pipe.to(device); student_pipe.text_encoder.to("cpu"); student_pipe.vae.to("cpu")
    load_quantized_transformer(student_pipe, CKPT, MODEL)
    student = student_pipe.transformer; freeze_all(student)
    student.enable_gradient_checkpointing()
    make_autograd_safe(student)
    anchor_path = run / "base_svdquant_anchor.pt"
    anchor_weights = load_base_anchor(student, anchor_path) if anchor_path.exists() else None
    dynamic = DynamicSmooth(student, teacher_pipe.transformer, CKPT, bound=1.25,
                            anchor_weights=anchor_weights).to(device)
    dynamic.teacher = None; teacher_pipe.to("cpu"); del teacher_pipe; gc.collect(); torch.cuda.empty_cache()
    originals, _ = install_activation_ste(student)
    rows = []
    try:
        for epoch in epochs:
            load_joint_state(student, dynamic, run / f"epoch_{epoch:02d}.pt")
            aggregate, per_sample = evaluate(student, records, device)
            rows.append({"epoch": epoch, **aggregate})
            print(json.dumps(rows[-1]), flush=True)
            for row in per_sample: row["epoch"] = epoch
        suffix = f"heldout_first{args.limit_items}" if args.limit_items else "heldout_epoch_curve"
        with (run / f"{suffix}.json").open("w") as handle: json.dump(rows, handle, indent=2)
        with (run / f"{suffix}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    finally:
        restore_activation_processors(originals)


if __name__ == "__main__":
    main()
