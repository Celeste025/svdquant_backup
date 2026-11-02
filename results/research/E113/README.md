# E113 — MiniMax-H3 mixed NVFP4/FP8 attention generation

**Numerical comparison update (E115):** The E113 Triton/official FP4 gaps
below predate the E115 E2M1 midpoint and K/V traversal-order fixes. See
`results/research/E115/README.md` for corrected comparisons; do not interpret
the historical cross-implementation NMSE as a mixed-precision routing effect.

Status: complete (2026-10-20). All six generated media, both paired-analysis
reports, and the independent `verify_e113_native.py --require-analysis` audit
pass.

## Fixed experiment

- Cases: race, prompt 16 / seed 20026; clap, prompt 161 / seed 38001610.
- 576×1024, 124 frames, 24 FPS, 20 denoising steps, CFG 1, video/audio flow
  shifts 12/3, CPU-seeded initial noise. All 200 target linears compute in BF16.
- Attention arms: official SageAttention3 (`sage3`), a native Triton all-NVFP4
  control (`native_fp4`), and the same native Triton core with V-error-guided
  FP8 P×V fallback (`native_mixed`). The router threshold is V tile FP4
  relative L2 > 0.10, per head and 128-key tile.
- Native Q×K and FP4 P×V use E2M1 values with E4M3 scale per 16 reduction
  elements. FP8 P×V uses E4M3 values with one dynamic FP32 scale per 128-key
  P row or V output channel. Q/K/V preprocessing follows the official Sage3
  `preprocess_qkv(..., per_block_mean=True)` call.

BF16 controls are pre-existing, not regenerated: the race video's
`results/samples/minimax_h3_svdquant_standard_8p64s/p16/bf16.json` records
prompt 16, seed 20026 and the exact size/step count; its inference call uses
CFG 1, CPU noise and tiled decode, with pipeline-default shifts 12/3. The
clap BF16 manifest `results/research/E073/denoise_full_bf16_r0.json` records
prompt 161, seed 38001610, all size/step settings and the shifts explicitly.
For clap, the new Sage3 initial video and audio latent SHA256 hashes also match
that historical BF16 manifest exactly. The race BF16 record does not expose an
initial-latent hash, so for race only prompt/seed/settings are independently
checked against the old BF16 source; the three new attention arms do have
matching initial-latent hashes.

The meaningful routing comparison is `native_mixed` against `native_fp4`.
Comparing only with official Sage3 also includes differences between attention
kernel implementations. Full-generation decoded RGB NMSE and LPIPS are paired
distances to the same-seed BF16 video, **not** teacher-forced DiT-output NMSE
or a direct perceptual quality score. The two cases do not support broad
generalization.

## Numerical gate and sources

- [Native Triton versus PyTorch FP8-group-128 validation](native_vs_pytorch_group128_full.json):
  13 real-input settings pass with finite outputs and NMSE < 1e-5. At one
  9,482-token, head-4 sequence, threshold 0.10 routes 3/75 V tiles to FP8,
  and output NMSE versus PyTorch is 1.1232e-6. Generated PTX contains native
  NVFP4 block-scale and FP8 MMA instructions.
- [Real-attention preflight](native_full_attention_preflight.json) measures one
  full H3 shape before generation. The correctness-first native Triton core is
  substantially slower than the optimized Sage3 kernel; its generation run is
  a numerical/visual experiment, not a deployment-speed claim.
- Scripts: `scripts/research/e113_native_mixed_attention.py`,
  `scripts/research/run_e113_h3_mixed_pv_video.py`,
  `scripts/research/summarize_e113_native_videos.py`, and
  `scripts/research/evaluate_e113_native_lpips.py`.

## Paired generation result

All values below are 124 decoded RGB frames, paired to the historical BF16
clip. Lower is closer to that clip; neither metric alone establishes visual
quality. The new attention arms share seed, settings, and initial video/audio
latent hashes. The race BF16 clip has no historical initial-latent hash, as
noted above.

| Case | Attention arm | RGB NMSE vs BF16 | LPIPS-Alex vs BF16 |
| --- | --- | ---: | ---: |
| Race | Sage3 FP4 | 0.17682 | 0.32612 |
| Race | Native all-FP4 | 0.16597 | 0.31497 |
| Race | Native mixed FP4/FP8 | 0.17030 | 0.31819 |
| Clap | Sage3 FP4 | 0.62548 | 0.48468 |
| Clap | Native all-FP4 | 0.66988 | 0.47934 |
| Clap | Native mixed FP4/FP8 | 0.67982 | 0.50536 |

Relative to the matched native all-FP4 control, the mixed route's RGB NMSE
increased by 2.61% (race) and 1.48% (clap); LPIPS increased by 1.02% and
5.43%, respectively. The mixed-vs-all-FP4 RGB NMSE is 0.10129 (race) and
0.31146 (clap), showing that the routing changes the free trajectory even
though its one-call attention-output NMSE advantage was small. In the poster
frames, race preserves the same general scene but changes some details and
timing; the clap variants diverge in viewpoint from the BF16 reference.
These are only two fixed cases, and the current V-tile error threshold does
not demonstrate a consistent generation-level gain. Do not compare mixed to
Sage3 alone to attribute a routing effect: the native FP4 core itself differs
from official Sage3.

Per-case sources: [race RGB summary](race/four_arm_attention_summary.json),
[race LPIPS](race/four_arm_attention_lpips.json),
[clap RGB summary](clap0/four_arm_attention_summary.json), and
[clap LPIPS](clap0/four_arm_attention_lpips.json). The four-arm montages are
`/data1/models/svdquant-wjq/research/20261020/E113_h3_mixed_pv_generation/`
under each case directory; the synchronized montages are silent, while the
original per-arm MP4s retain generated audio.

Run `scripts/research/verify_e113_native.py --require-analysis` for the final
cross-case audit. It independently rechecks numeric MMA validation, all six
denoise/video manifests, source and output hashes, decoded frame/audio
contracts, and RGB/LPIPS report linkage and finite values.

