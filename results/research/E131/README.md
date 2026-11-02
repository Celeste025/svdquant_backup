# E131 — race, 10% random-token BF16 recompute after native NVFP4 W4A4

Case p016, seed 20026, 576×1024, 124 frames, 20 denoising steps. Baselines:
E124 BF16 and E128 native NVFP4 W4A4 with BF16 attention. All three initial
video/audio latent hashes match. Only the 200 target main-block linears change;
attention and other modules remain BF16. At each linear call, a fresh exact-size
10% sample of token rows is drawn without replacement. The full linear input and
original weight are BF16 for those rows, and their BF16 `F.linear` outputs replace
the corresponding native W4A4 outputs. Each layer has a distinct seeded CUDA RNG
whose state advances every denoising step. The original BF16 weights stay resident
for this experiment, so memory/speed are not deployment measurements.

Preflight passed at 51.08 GiB peak GPU memory. Denoising completed 20 steps;
all 200 layers recomputed 20 times, 9,088,000 token rows in total. Video decoded
to 124 frames. Full rollout decoded-RGB metrics (smaller is closer to BF16):

| Arm vs BF16 | RGB NMSE | LPIPS-Alex |
|---|---:|---:|
| Native W4A4 | 0.298725 | 0.466281 |
| W4A4 + 10% token recompute | 0.266195 | 0.434069 |

Relative reductions: 10.89% RGB NMSE and 6.91% LPIPS. This is one prompt and
one random-mask seed; neither metric by itself proves perceptual improvement.
The same experiment was then run on VBench 000, "a person swimming in ocean",
seed 42, 768×1344, 124 frames and 50 denoising steps. Its BF16, W4A4 and
recompute runs also have identical initial video/audio latent hashes. All 200
linears recomputed on every one of the 50 steps, totaling 37,760,000 rows.

| Swimming arm vs BF16 | RGB NMSE | LPIPS-Alex |
|---|---:|---:|
| Native W4A4 | 0.102342 | 0.362326 |
| W4A4 + 10% token recompute | 0.094494 | 0.346306 |

Relative reductions: 7.67% RGB NMSE and 4.42% LPIPS. Both cases improve on
these frame-paired metrics, but two prompts are not enough to characterize the
distribution of benefits. Detailed metrics and paths are in
`race_video_metrics.json` and `vbench_000_video_metrics.json`.
