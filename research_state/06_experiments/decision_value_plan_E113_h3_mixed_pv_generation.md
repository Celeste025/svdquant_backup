# E113 — BF16-linear H3 Sage3 versus V-guided mixed P/V generation

Experiment ID: E113. Mode: exploration.

Problem addressed: The E112 one-block attention NMSE result may not predict
20-step free-generation quality. Test whether V-guided mixed P/V precision
actually changes the fixed race and clap outputs when linear layers remain
BF16.

Hypothesis: At the repository's 0.10 V-FP4-relative-L2 threshold, mixed
attention may improve generated output relative to pure Sage3, but the E112
small local gain makes a large effect uncertain.

Decision this informs: Whether the current V-error router merits a deployable
native mixed-bit kernel or a different selection criterion is needed.

Minimum valid setup: Preserve race prompt 16/seed 20026 and clap prompt
161/seed 38001610; 576x1024, 124 frames, 20 steps, CFG 1, video/audio
flow shifts 12/3, CPU noise. Keep every H3 linear BF16. Use the same Sage3
NVFP4 Q/K semantics; compare pure Sage3 to per-(head,128-key-tile) P/V FP4
or FP8 routing at threshold 0.10. Generate paired free trajectories and
validate media, then compare both to the existing BF16 control. Reuse an
existing pure-Sage3 clip only after confirming prompt/seed/noise/settings.

2026-10-20 amendment: The first PyTorch/Triton emulation used BF16 GEMMs and
was not a native precision implementation. The run now requires packed E2M1
plus E4M3-per-16 native NVFP4 MMA for QK and FP4 PV, with native E4M3 FP8
MMA for the selected PV tiles. Both native Triton and its PyTorch reference
use FP32 dynamic FP8 scales over 128 K-elements (one per P query row and one
per V output channel); the NVFP4 tile-routing criterion remains the per-16
V-FP4 relative L2 error at threshold 0.10. Because all-FP4 Triton differs
numerically from official Sage3 even before routing, include an explicit
`native_fp4` control alongside `sage3` and `native_mixed`. The `native_mixed`
vs `native_fp4` comparison isolates the routing effect; `native_mixed` vs
`sage3` alone does not.

Invalid cheap proxy to avoid: E112's one-block output NMSE or teacher-forced
DiT replay presented as generation quality; silently changing to per-head
Sage3/Sage2++ routing; quantizing linear layers.

Expected outcomes: Mixed videos either visibly/quantitatively improve, remain
similar, or worsen relative to pure Sage3. Report error magnitude and visual
examples for each case without a two-case generalization.

Decision for each outcome: Continue the current V router only for reproducible
improvement at tolerable runtime; otherwise revisit P/PV-impact routing.

Stop condition: Benchmark one real H3 attention call first. If extrapolated
runtime or memory makes four 20-step trajectories infeasible on available GPU
without disrupting other jobs, stop before launching long generation and
document the measured blocker. Do not silently substitute another method.

Evidence: Necessary — checked precision/router identity and paired free-rollout
videos. Supporting — per-step attention route fractions and video distance to
BF16. Cuttable — broad benchmark sweep.

2026-10-20 numerical gate: `scripts/research/validate_e113_native_mixed.py
--full` passed on the real E098 H3 QKV capture, including one full 9,482-token
head at the 0.10 threshold. The native Triton output versus the PyTorch
FP8-group-128 reference has NMSE 1.1232e-6; 72/75 V tiles selected FP4, so
the FP8 path was exercised on 3 tiles. All 13 validation settings are finite,
the maximum NMSE is 3.2178e-6, and PTX contains both NVFP4 block-scale and
FP8 MMA. Results: `results/research/E113/native_vs_pytorch_group128_full.json`.
