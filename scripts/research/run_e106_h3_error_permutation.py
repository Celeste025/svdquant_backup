#!/usr/bin/env python3
"""E106: permute genuine paired H3 DiT-output error on each arm's own rollout."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import run_e105_h3_output_interventions as base


ARM = "permuted_error"
base.ARMS = (ARM,)
base.DATA = Path("/data1/models/svdquant-wjq/research/20261017/E106_h3_error_permutation")
base.REPORT = base.ROOT / "results/research/E106"
base.PLAN = base.ROOT / "research_state/06_experiments/decision_value_plan_E106_h3_error_permutation.md"
base.DENOISE_FORMAT = "h3-e106-error-permutation-denoise-v1"
base.VIDEO_FORMAT = "h3-e106-error-permutation-video-v1"


def permute_error(reference: torch.Tensor, quantized: torch.Tensor, arm: str,
                  seed: int, step: int, modality_id: int) -> tuple[torch.Tensor, dict]:
    base.require(arm == ARM and reference.shape == quantized.shape and
                 reference.dtype == quantized.dtype == torch.bfloat16,
                 "paired outputs must have matching BF16 shape and dtype")
    ref = reference.float()
    error = quantized.float() - ref
    target_mse = float(error.square().mean(dtype=torch.float64))
    ref_energy = float(ref.square().mean(dtype=torch.float64))
    base.require(target_mse > 0 and math.isfinite(target_mse) and ref_energy > 0 and
                 bool(torch.isfinite(error).all()), "invalid paired output error")
    random_seed = seed + 30_000_000 + step * 1000 + modality_id
    generator = torch.Generator(device=reference.device).manual_seed(random_seed)
    permutation = torch.randperm(error.numel(), generator=generator, device=reference.device)
    intended_error = error.reshape(-1)[permutation].reshape_as(error)
    # The FP32 multiset is exactly preserved here. Casting back to the original
    # BF16 model-output dtype is mandatory for a fair downstream rollout.
    modified = (ref + intended_error).to(torch.bfloat16)
    realized_error = modified.float() - ref
    realized_mse, realized_nmse = base.estimate(modified, reference)
    distortion = float((realized_error - intended_error).square().mean(dtype=torch.float64) / target_mse)
    raw_cosine = float((error * intended_error).sum(dtype=torch.float64) /
                       error.square().sum(dtype=torch.float64))
    row = {
        "target_mse": target_mse,
        "target_nmse": target_mse / ref_energy,
        "reference_mean_square": ref_energy,
        "output_elements": error.numel(),
        "random_seed": random_seed,
        "permutation_scope": "all scalar elements within the modality output",
        "fp32_values_preserved_before_bf16_cast": True,
        "original_vs_permuted_same_position_cosine": raw_cosine,
        "bf16_rounding_distortion_over_target_mse": distortion,
        "realized_mse": realized_mse,
        "realized_nmse": realized_nmse,
        "relative_mse_change": realized_mse / target_mse - 1,
    }
    base.require(bool(torch.isfinite(modified).all()) and
                 abs(row["relative_mse_change"]) <= 0.005 and distortion <= 0.005,
                 f"permutation or BF16 rounding exceeded numerical gate: {row}")
    return modified, row


base.alter = permute_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--case", choices=tuple(base.CASES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    if args.phase == "preflight":
        base.preflight(args.case)
    elif args.phase == "denoise":
        base.denoise(args.case, ARM)
    else:
        base.decode(args.case, ARM)


if __name__ == "__main__":
    main()
