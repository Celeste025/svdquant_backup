#!/usr/bin/env python3
"""E137: isolate video-output error spatial versus temporal placement on H3 rollouts."""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import torch

import run_e105_h3_output_interventions as base


ARMS = ("spatial_shared", "temporal_frame")
VIDEO_GRID_THW = (37, 18, 32)
ATTEMPT = os.environ.get("E137_ATTEMPT", "attempt2")
# A paired run may be safely stopped if another user's process consumes the
# reserved memory.  Isolate retries, rather than resuming a partial trajectory.
if not re.fullmatch(r"attempt[0-9A-Za-z_-]+", ATTEMPT):
    raise RuntimeError("E137_ATTEMPT must be a path-safe retry identifier")
base.ARMS = ARMS
base.DATA = Path("/data1/models/svdquant-wjq/research/20261030/E137_h3_error_axis_permutation") / ATTEMPT
base.REPORT = base.ROOT / "results/research/E137" / ATTEMPT
base.PLAN = base.ROOT / "research_state/06_experiments/decision_value_plan_E137_h3_error_axis_permutation.md"
base.DENOISE_FORMAT = "h3-e137-axis-permutation-denoise-v1"
base.VIDEO_FORMAT = "h3-e137-axis-permutation-video-v1"


def axis_permute_error(reference: torch.Tensor, quantized: torch.Tensor, arm: str,
                       seed: int, step: int, modality_id: int) -> tuple[torch.Tensor, dict]:
    base.require(arm in ARMS and reference.shape == quantized.shape and
                 reference.dtype == quantized.dtype == torch.bfloat16,
                 "paired outputs must have matching BF16 shape and dtype")
    ref = reference.float()
    error = quantized.float() - ref
    target_mse = float(error.square().mean(dtype=torch.float64))
    ref_energy = float(ref.square().mean(dtype=torch.float64))
    base.require(target_mse > 0 and math.isfinite(target_mse) and ref_energy > 0 and
                 bool(torch.isfinite(error).all()), "invalid paired output error")

    row = {
        "target_mse": target_mse,
        "target_nmse": target_mse / ref_energy,
        "reference_mean_square": ref_energy,
        "output_elements": error.numel(),
        "intervention_scope": "video_only",
    }
    if modality_id == 1:
        modified = quantized
        row.update(axis="none", reason="audio error is deliberately unchanged")
    else:
        time, height, width = VIDEO_GRID_THW
        base.require(error.ndim == 2 and tuple(error.shape) == (time * height * width, 96),
                     f"unexpected video output layout for {VIDEO_GRID_THW}: {tuple(error.shape)}")
        generator = torch.Generator(device=reference.device).manual_seed(
            seed + (137_000_000 if arm == "spatial_shared" else 138_000_000) + step
        )
        structured = error.reshape(time, height, width, 96)
        if arm == "spatial_shared":
            permutation = torch.randperm(height * width, generator=generator, device=reference.device)
            intended_error = structured.reshape(time, height * width, 96)[:, permutation, :].reshape_as(error)
            row.update(axis="spatial", permutation_scope="one HxW token permutation shared by all T frames",
                       permutation_length=height * width)
        else:
            permutation = torch.randperm(time, generator=generator, device=reference.device)
            intended_error = structured[permutation].reshape_as(error)
            row.update(axis="temporal", permutation_scope="one whole-frame permutation shared by all HxW locations",
                       permutation_length=time)
        modified = (ref + intended_error).to(torch.bfloat16)
        raw_cosine = float((error * intended_error).sum(dtype=torch.float64) /
                           error.square().sum(dtype=torch.float64))
        row.update(random_seed=int(generator.initial_seed()),
                   fp32_token_error_vectors_preserved_before_bf16_cast=True,
                   original_vs_relocated_same_position_cosine=raw_cosine,
                   bf16_rounding_distortion_over_target_mse=float(
                       ((modified.float() - ref - intended_error).square().mean(dtype=torch.float64)) / target_mse))

    realized_mse, realized_nmse = base.estimate(modified, reference)
    row.update(realized_mse=realized_mse, realized_nmse=realized_nmse,
               relative_mse_change=realized_mse / target_mse - 1)
    base.require(bool(torch.isfinite(modified).all()) and abs(row["relative_mse_change"]) <= 0.005,
                 f"axis permutation or BF16 rounding exceeded numerical gate: {row}")
    if modality_id == 0:
        base.require(row["bf16_rounding_distortion_over_target_mse"] <= 0.005,
                     f"axis permutation rounding distortion exceeded gate: {row}")
    return modified, row


base.alter = axis_permute_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--case", choices=tuple(base.CASES), required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    base.require((args.phase == "preflight") == (args.arm is None),
                 "--arm is required only for denoise/decode")
    torch.set_num_threads(6)
    if args.phase == "preflight":
        base.preflight(args.case)
    elif args.phase == "denoise":
        base.denoise(args.case, args.arm)
    else:
        base.decode(args.case, args.arm)


if __name__ == "__main__":
    main()
