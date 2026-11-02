#!/usr/bin/env python3
"""E108: closed-loop H3 correction using the previous true paired output error."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import run_e105_h3_output_interventions as base


ARM = "oracle_prev_error_k06"
K = 0.6
base.ARMS = (ARM,)
base.DATA = Path("/data1/models/svdquant-wjq/research/20261017/E108_h3_oracle_temporal_correction")
base.REPORT = base.ROOT / "results/research/E108"
base.PLAN = base.ROOT / "research_state/06_experiments/decision_value_plan_E108_h3_oracle_temporal_correction.md"
base.DENOISE_FORMAT = "h3-e108-oracle-previous-error-denoise-v1"
base.VIDEO_FORMAT = "h3-e108-oracle-previous-error-video-v1"

# One process handles exactly one case. Each entry contains the *uncorrected*
# Q-BF16 error at the preceding step for one modality.
previous: dict[int, tuple[int, torch.Tensor]] = {}


def correct(reference: torch.Tensor, quantized: torch.Tensor, arm: str,
            seed: int, step: int, modality_id: int) -> tuple[torch.Tensor, dict]:
    base.require(arm == ARM and reference.shape == quantized.shape and
                 reference.dtype == quantized.dtype == torch.bfloat16,
                 "paired outputs must have matching BF16 shape and dtype")
    ref = reference.float()
    q = quantized.float()
    error = q - ref
    target_mse = float(error.square().mean(dtype=torch.float64))
    ref_energy = float(ref.square().mean(dtype=torch.float64))
    base.require(target_mse > 0 and ref_energy > 0 and math.isfinite(target_mse) and
                 bool(torch.isfinite(error).all()), "nonfinite paired output error")
    row = {"target_mse": target_mse, "target_nmse": target_mse / ref_energy,
           "reference_mean_square": ref_energy, "output_elements": error.numel(),
           "coefficient": K, "previous_error_definition": "raw previous Q-BF16 on this arm's own trajectory"}
    if step == 0:
        base.require(modality_id not in previous, "unexpected prior error at step 0")
        modified = quantized
        row.update(previous_error_available=False, prior_error_cosine=None,
                   previous_over_current_error_rms=None, k_opt_oracle=None,
                   bf16_rounding_distortion_over_target_mse=0.0)
    else:
        base.require(modality_id in previous and previous[modality_id][0] == step - 1,
                     "missing or misaligned preceding true error")
        prior = previous[modality_id][1]
        base.require(prior.shape == error.shape, "previous error shape mismatch")
        intended = q - K * prior
        modified = intended.to(torch.bfloat16)
        prior_energy = float(prior.square().sum(dtype=torch.float64))
        current_energy = float(error.square().sum(dtype=torch.float64))
        cross = float((error * prior).sum(dtype=torch.float64))
        row.update(previous_error_available=True,
                   prior_error_cosine=cross / math.sqrt(current_energy * prior_energy),
                   previous_over_current_error_rms=math.sqrt(prior_energy / current_energy),
                   k_opt_oracle=cross / prior_energy,
                   bf16_rounding_distortion_over_target_mse=float(
                       (modified.float() - intended).square().mean(dtype=torch.float64) / target_mse))
        base.require(row["bf16_rounding_distortion_over_target_mse"] <= 0.01,
                     "BF16 correction rounding distortion exceeded 1% of raw error energy")
    realized_mse, realized_nmse = base.estimate(modified, reference)
    row.update(realized_mse=realized_mse, realized_nmse=realized_nmse,
               relative_mse_change=realized_mse / target_mse - 1)
    base.require(bool(torch.isfinite(modified).all()) and math.isfinite(realized_nmse),
                 "nonfinite corrected output")
    if step == 0:
        base.require(torch.equal(modified, quantized) and realized_mse == target_mse,
                     "step 0 must exactly reproduce original SVDQuant")
    previous[modality_id] = (step, error.detach().clone())
    return modified, row


base.alter = correct


def preflight(case: str) -> None:
    previous.clear()
    base.preflight(case)
    previous.clear()
    # Validate the stateful rule independently of model loading on two steps.
    reference0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    quantized0 = torch.tensor([[1.125, 1.875], [3.25, 3.75]], dtype=torch.bfloat16)
    output0, _ = correct(reference0, quantized0, ARM, 0, 0, 0)
    reference1 = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.bfloat16)
    quantized1 = torch.tensor([[1.75, 2.25], [3.75, 4.25]], dtype=torch.bfloat16)
    output1, row1 = correct(reference1, quantized1, ARM, 0, 1, 0)
    expected = (quantized1.float() - K * (quantized0.float() - reference0.float())).to(torch.bfloat16)
    base.require(torch.equal(output0, quantized0) and torch.equal(output1, expected) and
                 row1["previous_error_available"], "stateful correction preflight failed")
    previous.clear()
    print(f"ORACLE PREVIOUS-ERROR PREFLIGHT PASS {case}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--case", choices=tuple(base.CASES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    if args.phase == "preflight":
        preflight(args.case)
    elif args.phase == "denoise":
        previous.clear()
        base.denoise(args.case, ARM)
    else:
        base.decode(args.case, ARM)


if __name__ == "__main__":
    main()
