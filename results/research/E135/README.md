# E135: BF16 at both ends, native NVFP4 W4A4 in the middle

## Setup

For race `p016` and swimming `vbench_000`, the first and last 10% of DiT
denoising steps use the original BF16 model. The middle 80% use the existing
*native* NVFP4 E2M1 W4A4 implementation: 200 original-weight linears, group-16
block scale, dynamic activation tensor-wise scale, no smoothing or low-rank
branch. Its packed weight export is exactly the E128 pure-W4A4 export (manifest
SHA256 `687a8a0bcb2d551820b287797c16e98759ae487969cd5efe512f0f847cbdf99f`).
Attention remains BF16/torch in all arms. This is a closed-loop switch of the
full DiT output at each timestep, not a post-hoc blend.

| Case | Seed | Output | Steps | BF16 steps (1-based) | Native W4A4 steps |
|---|---:|---|---:|---|---|
| Race | 20026 | 576×1024, 124 frames | 20 | 1–2, 19–20 | 3–18 |
| Swimming | 42 | 768×1344, 124 frames | 50 | 1–5, 46–50 | 6–45 |

The reference BF16 videos are E124/E126; the pure native-W4A4 videos are E128;
the rank-64 SVDQuant video is E129; and the BF16-ends/SVDQuant-middle videos
are E134. All compared arms reuse identical prompt, seed, scheduler and initial
video/audio latents. The evaluator verifies the initial latent hashes.

## Results

Metrics are computed against the decoded full-BF16 video over all 124 paired
RGB frames. RGB NMSE is whole-video sum-squared error divided by BF16 RGB
energy; LPIPS is the frame-average AlexNet LPIPS. Lower is closer to BF16.

| Case | Arm | RGB NMSE | LPIPS |
|---|---|---:|---:|
| Race | Pure native W4A4 | 0.298725 | 0.466281 |
| Race | BF16 ends + native W4A4 middle | 0.296440 | 0.439765 |
| Race | Pure SVDQuant | 0.254097 | 0.423326 |
| Race | BF16 ends + SVDQuant middle | 0.228127 | 0.373093 |
| Swimming | Pure native W4A4 | 0.102342 | 0.362326 |
| Swimming | BF16 ends + native W4A4 middle | 0.069521 | 0.292571 |
| Swimming | Pure SVDQuant | 0.107454 | 0.421154 |
| Swimming | BF16 ends + SVDQuant middle | 0.041854 | 0.227845 |

Relative to pure native W4A4, switching the end windows to BF16 improves RGB
NMSE by 0.77% and LPIPS by 5.69% for race, and RGB NMSE by 32.07% and LPIPS by
19.25% for swimming. Both mixed-W4A4 results are less close to BF16 than the
corresponding mixed-SVDQuant results. However, the W4A4 arms use a packed native
FP4 GEMM while E129/E134 SVDQuant arms use pseudo-NVFP4 Q/DQ with BF16 GEMM.
The comparison therefore cannot isolate the effect of the low-rank branch;
quantization implementation and rounding are also different. The two-case
experiment is not a population-level quality claim; closeness to BF16 is not
a perceptual preference or VBench score.

## Artifacts

- `p016_video_metrics.json` and `vbench_000_video_metrics.json`: all four
  non-BF16 arms evaluated against the same BF16 reference, with source paths
  and initial latent hashes.
- `{case}/bf16_both10pct_native_w4a4_denoise.json`: actual per-step precision,
  timings, output energies, packed-weight manifest hash, seed and latent hashes.
- `{case}/bf16_both10pct_native_w4a4_video.json`: decoded video metadata.
- Generation: `scripts/research/run_e135_h3_bf16_w4a4_window.py` and
  `scripts/research/run_e135_case.sh`.
- Evaluation: `scripts/research/evaluate_e135_videos.py`.
- Logs: `results/logs/e135_race.log`, `results/logs/e135_swim.log`.
- Large videos and final latents:
  `/data1/models/svdquant-wjq/research/20261028/E135_h3_bf16_native_w4a4_window/`.
  Preview copies: `results/media/E135/`.

Run one case with `bash scripts/research/run_e135_case.sh GPU p016` (or
`vbench_000`). Existing results are not overwritten. Recompute metrics with
`CUDA_VISIBLE_DEVICES=GPU /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/evaluate_e135_videos.py --case CASE`.
