# E133 — 10% DiT-output token restoration at every denoising step

At every step, the same current input is sent through one GPU-resident DiT
twice: once with original BF16 weights and once with E128 native NVFP4 W4A4
for its 200 target linears. Attention and non-target layers remain BF16.
For each video and audio output tensor separately, 10% of token rows in the
W4A4 output are replaced by the corresponding BF16 rows. The mixed output is
then passed to the scheduler, so all later steps see the intervened trajectory.

`random10` uses a fresh seeded, without-replacement sample per step and
modality. `cycle10` divides token rows in their original order into ten
contiguous partitions, restores partition `step % 10`, and repeats after ten
steps. Thus every token is restored once per ten steps. This deliberately
tests a sequential policy; it is not spatially interleaved.

| Case / arm | RGB NMSE vs BF16 | LPIPS-Alex vs BF16 |
|---|---:|---:|
| Race: pure W4A4 | 0.298725 | 0.466281 |
| Race: random10 | 0.288951 | 0.442033 |
| Race: cycle10 | 0.273484 | 0.437365 |
| Swimming: pure W4A4 | 0.102342 | 0.362326 |
| Swimming: random10 | 0.096822 | 0.353342 |
| Swimming: cycle10 | 0.096507 | 0.346553 |

Relative to pure W4A4, race RGB NMSE falls 3.27% with random10 and 8.45%
with cycle10; swimming falls 5.39% and 5.70%. All four runs share their
case's exact initial video/audio latent hashes with BF16 and W4A4 baselines.
At the intervened DiT output, the selected 10% rows remove about 10% of
the contemporaneous error energy on average, as expected for a token sample.

These are two prompts and one random seed per case. Frame-paired RGB NMSE
and LPIPS quantify closeness to independently generated BF16 videos, not a
human quality score. The two policies differ in spatial clustering as well
as temporal coverage, so cycle10's better result here does not establish
that deterministic traversal is intrinsically superior.

Full per-step modality NMSE and replaced-token counts are in each arm's
`*_denoise.json`; video-level metrics and paired source paths are in
`p016_video_metrics.json` and `vbench_000_video_metrics.json`.
