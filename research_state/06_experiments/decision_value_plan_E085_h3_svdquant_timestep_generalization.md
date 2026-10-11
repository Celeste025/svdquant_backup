# E085 — MiniMax-H3 SVDQuant timestep NMSE generalization

## Experiment ID

E085

## Problem addressed

Determine whether the apparent timestep-dependent SVDQuant NMSE pattern is a
general property or a calibration-set artifact in the standard MiniMax-H3
W4A4/NVFP4 SVDQuant state.

## Hypotheses

1. Teacher-forced DiT-output NMSE declines as denoising advances.
2. The first sampled timestep receives a disproportionately large SVDQuant
   improvement relative to later timesteps.
3. The direction and concentration of either effect are shared by calibration
   and held-out prompts.

## Decision informed

Decide whether timestep-specific calibration/method changes are warranted, or
whether the observation is too sample-/split-dependent to motivate them.

## Minimum valid setup

Use the immutable `minimax_h3_svdquant_standard_8p64s` state.  Compare the
full BF16 and SVDQuant DiT outputs under identical, BF16-captured call inputs:
20 calls for each held-out prompt and the existing 64 cached calibration
calls.  Evaluate video and audio separately.  Held-out prompts are the five
pre-existing standard samples {2, 16, 26, 42, 51}; calibration prompts are
{1, 11, 20, 25, 46, 48, 105, 116}.

## Invalid cheap proxy to avoid

Do not compare free-running SVDQuant and BF16 at later steps: that confounds
local quantization error with divergent latents.  Do not infer held-out
behavior from one prompt.

## Expected outcomes and action

- Shared monotonic/early-step pattern: investigate timestep-weighted
  calibration with a prespecified held-out gate.
- Calibration-only or materially heterogeneous pattern: treat the original
  observation as descriptive, not as a method-design premise.
- No directional pattern: stop timestep-specific follow-up.

## Stop condition

Stop after all five held-out prompt trajectories are evaluated and aggregate
statistics include uncertainty across prompts.  No search or retraining is
authorized by this experiment.
