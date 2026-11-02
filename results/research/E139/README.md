# E139 — DraftAttention2 generation preview (completed 2026-10-31)

Official Anemoi source commit `97a1c8b14f26f387214c8cfb8fc7e890390b93b4`
in `third_party/anemoi-code`. Filtered clone avoids demonstration media.
Two build-only compatibility patches are exported in
`official_build_compatibility.patch`: NVCC-level cassert preinclude and an
explicitly typed tensor-check array in the host wrapper. Core attention
arithmetic, routing, quantization and precision policy remain official code.
Initial missing CUDA includes and compilation failures are retained in logs.

## Contract

BF16 H3 linear layers; BF16 refiner and independent padding segment. Race
prompt16/seed20026; clap prompt161/seed38001610. Both 576x1024, 124 frames,
20 denoising steps, CFG1, video/audio flow shifts12/3, original initial noise.
Video grid37x18x32; observed prefix counts1365/race and419/clap, checked as
contiguous non-video prefix followed by video tokens.

Official Q64, maxpool_weight0.2, per-layer H3 NVFP4 calibration, prefix INT8.
Keep the official default dense-first10 steps and dense-first2 layers.
For steps10–19 (zero-based), blocks2–49:

- `s70_int8_30`: 70% skipped video-region pairs, 30% INT8.
- `s70_fp4_20_int8_10`: 70% skipped, 20% NVFP4, 10% INT8;
  retained fractions are NVFP4=2/3 and INT8=1/3.

Fractions describe routed video regions only. Warmup, prefix interactions,
refiner and padding are outside this budget. They do not describe a global
70% reduction of the whole 20-step model. INT8 path is INT8 QK + E4M3 PV.

## Validation

Small dense FP16 vs PyTorch SDPA NMSE5.72e-6. Both sparse paths finite.
Four experimental arms completed20 finite-output steps and124-frame media
with audio. Initial latent hashes match the selected BF16 references.
Race warmup first-step output is byte-identical to E121 BF16. Clap warmup
did not byte-match the historical E125 record, so a new same-environment
BF16 was run; its first-step output byte-matches both experimental arms.
First adapter attempt failed before step0 due prehook/context ordering;
records are retained under `failed_start1` and were not used for conclusions.

## Results

All124 full-resolution decoded RGB frames vs matched BF16; lower is closer.

| Case | Arm | RGB NMSE | LPIPS-Alex | Total DiT seconds | Mean sparse-step DiT seconds |
|---|---|---:|---:|---:|---:|
| Race | 70 skip /30 INT8 | 0.016243 | 0.086220 | 187.51 | 8.414 |
| Race | 70 skip /20 FP4 /10 INT8 | 0.019715 | 0.106335 | 187.84 | 8.405 |
| Clap | 70 skip /30 INT8 | 0.017714 | 0.060311 | 189.05 | 8.503 |
| Clap | 70 skip /20 FP4 /10 INT8 | 0.021480 | 0.067990 | 188.16 | 8.431 |

Timing is a single full-DiT rollout, excluding load/VAE but including the
existing pipeline's weight management, attention preprocessing and routing.
It is not an isolated attention benchmark or repeated speed measurement.
No clear whole-DiT timing gain from mixing is established. Both cases have
larger BF16-reference distances under mixing. This does not establish a
perceptual quality ranking or that this budget is optimal.

Media root:
`/data1/models/svdquant-wjq/research/20261031/E139_draftattention2`.
Each case has a labeled BF16/INT8/mixed comparison video,124frames, no audio;
individual arm videos retain audio. Posters sample frame63 only, not motion.
Per-case `video_metrics.json` contains sources, source hashes, step timings,
initial latent hashes and comparison paths. All generation and evaluation
processes ended; no polling queue is left running.

Next research step, if requested: compare budgets on native attention calls
with matched measured latency before optimizing scoring. The current preview
does not isolate selection-rule suboptimality from reduced arithmetic precision.

## Paired operator timing follow-up

Real race block25 QKV `[1,56,22677,128]`, video37x18x32 and prefix1365,
noncausal. GPU0 RTX PRO5000. Full API including native preprocess and, for
Anemoi, draft scoring/routing/packing/output assembly. HND/BSHD layouts and
baseline Sage3 K restore prepared before timing. Source E130 readonly Sage3,
actual device SM count110; optimized variant enables the four existing flags.
25 warmups plus72 calls in each of forward/reverse orders:144 measured calls
per arm, CUDA events, no trimming. Per-case rollout inputs differ across steps;
this is one actual captured call, not a 50-layer/20-step average.

| API | Mean ms | Speedup vs BF16 | Same-input output NMSE |
|---|---:|---:|---:|
| BF16 SDPA | 68.565 | 1.000 | 0 |
| Sage3 grid110 | 32.647 | 2.100 | 0.007039 |
| Sage3 optimized | 28.852 | 2.376 | 0.007039 |
| Draft 70skip/30INT8 | 21.975 | 3.120 | 0.014947 |
| Draft 70skip/20FP4/10INT8 | 20.762 | 3.302 | 0.016305 |

Mixed speedup1.572x over baseline Sage3,1.390x over optimized Sage3,
1.058x over sparse INT8. Thus near-equal full-DiT timings do not mean equal
attention times. Whole-DiT rollout timings include linear/weight-management
costs and half of steps use dense attention. See `attention_speed.json` for
all round timings and `scripts/research/bench_e139_attention.py` for source.
