# E136 — H3 timestep precision windows

Mode: exploratory/completeness comparison explicitly requested by the user.

## Common contract

Race `p016` uses seed 20026, 576×1024, 124 frames, 20 denoising steps;
swimming `vbench_000` uses seed 42, 768×1344, 124 frames, 50 steps. The
reference is each case's existing full-BF16 E124/E126 video. All arms must
match the reference's initial video and audio latent hashes and use the same
prompt, scheduler, VAE, and BF16 attention unless an arm explicitly specifies
Sage3. Compare 124 decoded frames with RGB NMSE and AlexNet LPIPS; report
per-step precision and finite-output checks. These two cases are exploratory,
not a population-level significance test.

## E136-A/B — one-sided BF16 windows

- Problem: Does early or late full-precision DiT computation matter more when
  all other steps use rank-64 SVDQuant W4A4?
- Hypothesis: early BF16 has larger final-video effect.
- Decision: whether a precision scheduler should prioritize initial steps.
- Minimum valid setup: 20/50-step closed-loop trajectories, first-only and
  last-only windows, existing E134 arms on both cases.
- Invalid cheap proxy: replacing final video pixels or scoring independent
  teacher-forced DiT calls in place of generation.
- Expected outcomes: early benefit, late benefit, both, or neither; if early
  dominates, prioritize early precision; otherwise do not.
- Stop condition: reuse completed E134 arms after validating hashes, schedules,
  videos, and metrics; do not repeat GPU runs with identical settings.
- Evidence: necessary to the requested comparison.

## E136-C — INT8 ends, SVDQuant middle

- Problem: Can less-expensive 8-bit endpoint steps retain much of BF16's
  benefit over all-rank-64-SVDQuant generation?
- Hypothesis: symmetric group-128 W8A8 endpoints improve proximity to BF16
  compared with pure SVDQuant, but less than BF16 endpoints.
- Decision: whether INT8 endpoint kernels warrant a deployment implementation.
- Minimum valid setup: 200 original-weight target linears at endpoints;
  per-output-row weight K-groups and per-token activation K-groups of 128,
  symmetric signed INT8 [-127,127], FP32 absmax/127 scale per group,
  round-to-nearest-even, then BF16 GEMM on Q/DQ operands. Middle steps use
  existing rank-64 SVDQuant W4A4 Q/DQ and BF16 attention. Both cases run in
  closed loop with full 20/50 steps.
- Invalid cheap proxy: INT8 only on one layer, only weight quantization,
  fewer denoising steps, or treating Q/DQ latency as native INT8 speed.
- Expected outcomes: substantial, small, or negative endpoint benefit;
  pursue a true INT8 kernel only if quality is useful.
- Stop condition: two valid paired videos and finite metrics; do not tune
  scale granularity or bit convention on these evaluation prompts.
- Evidence: necessary to the requested comparison.

## E136-D — BF16 ends, SVDQuant + Sage3 middle

- Problem: Does adding corrected Sage3 attention to the middle SVDQuant
  steps change the quality benefit of BF16 endpoint windows?
- Hypothesis: the attention approximation may add error but BF16 endpoints
  could suppress some of it.
- Decision: whether this combination merits broader prompt testing.
- Minimum valid setup: existing rank-64 SVDQuant linears, corrected Sage3
  main attention for middle steps only (BF16 refiner and padding as in E121),
  full BF16 DiT at endpoints, 20/50-step closed loop on both cases.
- Invalid cheap proxy: swapping attention after the DiT forward, using the
  historical Sage3 padding bug, or applying Sage3 in BF16 endpoint steps.
- Expected outcomes: improvement, parity, or degradation vs E134 BF16-ends
  SVDQuant/BF16-attention arm; retain the route only if acceptable.
- Stop condition: two paired videos, validated 50-main/2-refiner calls for
  every middle step, finite metrics and matching initial latent hashes.
- Evidence: necessary to the requested comparison.

No training or checkpoint selection is performed. The single W8A8 convention
above is a declared assumption because the request specified INT8 group-128
without an exact signed/scale convention. It must not be revised silently.
