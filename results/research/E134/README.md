# E134: BF16 at both ends of denoising with rank-64 H3 SVDQuant in the middle

## Question and design

For the matched race (`p016`) and swimming (`vbench_000`) cases, run the full
BF16 DiT in both the first and the last 10% of denoising steps. Run the existing
rank-64 SVDQuant DiT in the middle 80%. Additional first-only and last-only
arms serve as ablations. Each selected DiT output is
passed into the scheduler and affects subsequent steps (closed-loop generation).
This is **not** a post-hoc blend or an output-token replacement.

| Case | Seed | Shape | Steps | BF16 first 10% | BF16 last 10% |
|---|---:|---|---:|---|---|
| Race `p016` | 20026 | 576×1024, 124 frames | 20 | 1–2 | 19–20 |
| Swimming `vbench_000` | 42 | 768×1344, 124 frames | 50 | 1–5 | 46–50 |

The middle steps use `results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt`
(SHA256 `d199da3e5c89d248653d72b035e57cc505644075f269c6f44cc4c552e1f10f73`),
with dynamic activation tensor-wise scale, and BF16/torch attention. The
SVDQuant linear path is the existing pseudo-NVFP4 path, not a native FP4 GEMM.
The BF16 and SVDQuant pipelines use the same model base, prompt, seed, scheduler,
initial video/audio latents, and VAE decoding settings as the prior E124/E126
BF16 and E129 pure SVDQuant videos. The evaluator asserts identical initial
video/audio latent hashes across all four arms for each case.

## Results

Decoded RGB NMSE and AlexNet LPIPS are computed over all 124 paired RGB frames
against the corresponding full-BF16 generation. Lower is closer to full BF16.
"Change" is relative to the pure SVDQuant metric; positive means improvement.

| Case | Denoising arm | RGB NMSE | NMSE change | LPIPS | LPIPS change |
|---|---|---:|---:|---:|---:|
| Race | Pure SVDQuant | 0.254097 | — | 0.423326 | — |
| Race | BF16 first 2 | 0.227447 | +10.49% | 0.378459 | +10.60% |
| Race | BF16 last 2 | 0.255634 | −0.60% | 0.419434 | +0.92% |
| Swimming | Pure SVDQuant | 0.107454 | — | 0.421154 | — |
| Swimming | BF16 first 5 | 0.042429 | +60.51% | 0.234624 | +44.29% |
| Swimming | BF16 last 5 | 0.107105 | +0.32% | 0.415934 | +1.24% |
| Race | BF16 first and last 2 | 0.228127 | +10.22% | 0.373093 | +11.87% |
| Swimming | BF16 first and last 5 | 0.041854 | +61.05% | 0.227845 | +45.90% |

The both-ends arm is the primary requested comparison. Relative to pure
SVDQuant, it reduces RGB NMSE by 10.22% for race and 61.05% for swimming.
The first-only and last-only ablations show that early steps have much more
influence on final video similarity than late steps in these two prompts.
Adding the last window to the first-only arm changes race RGB NMSE by −0.30%
(slightly worse) and LPIPS by +1.42% (better); for swimming it improves RGB
NMSE by 1.35% and LPIPS by 2.89%. The effect size varies substantially between
prompts. This two-case comparison does not establish a population-level quality
improvement, and similarity to the BF16 reference is not a perceptual preference
or VBench score.

## Artifacts and reproduction

- `p016_video_metrics.json` and `vbench_000_video_metrics.json`: paired metric
  values, source paths and initial-latent hashes.
- `{case}/{arm}_denoise.json`: per-step precision, DiT timing and output energy,
  checkpoint hash, settings, seed and latent hashes.
- `{case}/{arm}_video.json`: decoded video path and media metadata.
- Generation code: `scripts/research/run_e134_svdquant_bf16_window.py` and
  `scripts/research/run_e134_case.sh`.
- Evaluation code: `scripts/research/evaluate_e134_videos.py`.
- Logs: `results/logs/e134_{race,swim}_{first,last}.log`.
- Generated videos and final latents are stored outside Git under
  `/data1/models/svdquant-wjq/research/20261028/E134_svdquant_bf16_step_window/`.
  Workspace preview copies are in `results/media/E134/`.

To regenerate one arm, run `bash scripts/research/run_e134_case.sh GPU CASE ARM`
where `CASE` is `p016` or `vbench_000`, and `ARM` is `bf16_both10pct`,
`bf16_first10pct` or `bf16_last10pct`. The script refuses to overwrite an existing arm. The two
metric files can be recomputed with
`CUDA_VISIBLE_DEVICES=GPU /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/evaluate_e134_videos.py --case CASE`.
