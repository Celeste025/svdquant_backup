# E136 — H3 timestep precision-window comparison

## Scope and implementation

Two cases: race `p016` (seed 20026, 576×1024, 124 frames, 20 steps) and
swimming `vbench_000` (seed 42, 768×1344, 124 frames, 50 steps). They use the
same prompt/seed, scheduler, VAE settings and initial video/audio latents as
the E124/E126 full-BF16 references and E129 pure rank-64 SVDQuant baseline.
Only the 200 target main-block linears are quantized; refiner linears remain
BF16. All arms use closed-loop generation, so each timestep's selected DiT
output feeds the next scheduler step.

The four requested arms are:

1. First 10% steps full BF16, remaining steps rank-64 SVDQuant W4A4 and BF16
   attention: completed E134 first-only arm, reused without rerun.
2. Last 10% steps full BF16, earlier steps rank-64 SVDQuant W4A4 and BF16
   attention: completed E134 last-only arm, reused without rerun.
3. First and last 10% steps symmetric W8A8 INT8 K-group-128 Q/DQ on the 200
   original-weight linears, BF16 GEMM and BF16 attention; middle 80% rank-64
   SVDQuant W4A4 and BF16 attention. Weight groups are per output row, input
   groups per token row, both along K. Scale is current group absmax/127 in
   FP32; codes are signed [-127,127]; rounding is nearest-even. This is a
   numerical INT8 proxy **not** a native INT8 deployment-speed measurement.
4. First and last 10% steps full BF16; middle 80% rank-64 SVDQuant W4A4
   linears plus the corrected Sage3 FP4 main-attention route. Refiner and
   independent padding attention segments remain BF16. The Sage3 router is
   closed before the final BF16 steps.

SVDQuant uses the unchanged rank-64 checkpoint
`results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt`
(SHA256 `d199da3e5c89d248653d72b035e57cc505644075f269c6f44cc4c552e1f10f73`).
The SVDQuant linear path remains NVFP4 Q/DQ followed by BF16 GEMM. No
checkpoint or quantizer calibration was selected on these two prompts.

## Results

Decoded RGB NMSE and mean AlexNet LPIPS are against the paired full-BF16 video
over all 124 frames; lower means closer to this reference, not necessarily
better subjective quality.

| Case | Arm | RGB NMSE | LPIPS |
|---|---|---:|---:|
| Race | Pure SVDQuant, BF16 attention | 0.254097 | 0.423326 |
| Race | First 10% BF16 | 0.227447 | 0.378459 |
| Race | Last 10% BF16 | 0.255634 | 0.419434 |
| Race | Both 10% BF16, SVDQuant middle | 0.228127 | 0.373093 |
| Race | Both 10% INT8 K128, SVDQuant middle | 0.243501 | 0.388736 |
| Race | Both 10% BF16, SVDQuant + Sage3 middle | 0.251439 | 0.402089 |
| Swimming | Pure SVDQuant, BF16 attention | 0.107454 | 0.421154 |
| Swimming | First 10% BF16 | 0.042429 | 0.234624 |
| Swimming | Last 10% BF16 | 0.107105 | 0.415934 |
| Swimming | Both 10% BF16, SVDQuant middle | 0.041854 | 0.227845 |
| Swimming | Both 10% INT8 K128, SVDQuant middle | 0.043004 | 0.225904 |
| Swimming | Both 10% BF16, SVDQuant + Sage3 middle | 0.050175 | 0.226363 |

Relative to pure SVDQuant, first-only BF16 reduces RGB NMSE by 10.49% in
race and 60.51% in swimming; last-only BF16 changes it by −0.60% and +0.32%.
Both-end INT8 reduces RGB NMSE by 4.17% and 59.98%. In swimming, INT8
endpoints are close to BF16 endpoints (0.043004 versus 0.041854 RGB NMSE),
but the race gap is larger (0.243501 versus 0.228127). Substituting corrected
Sage3 into the middle raises RGB NMSE relative to the matching BF16-attention
middle by 10.22% in race and 19.88% in swimming, even though both remain
below the pure-SVDQuant RGB NMSE. LPIPS rankings are not identical: swimming
INT8-end and Sage3-middle LPIPS values are slightly below the BF16-end,
BF16-attention value. With only two cases, those small differences should not
be treated as a robust subjective-quality advantage.

## Integrity and artifacts

The scorer asserts identical initial video/audio latent hashes across all
seven arms for each case, validates all 124 RGB frames and their dimensions,
then computes paired metrics. E136 denoise records retain the actual precision
of every step, finite-output checks and checkpoint hash; Sage3 steps also
record exactly 50 main FP4 and 2 refiner attention calls each.

- `p016_video_metrics.json`, `vbench_000_video_metrics.json`: raw paired scores
  and video source paths.
- `{case}/{arm}_denoise.json`, `{case}/{arm}_video.json`: step records and media
  metadata for the two new arms.
- Generation: `scripts/research/run_e136_h3_precision_windows.py` and
  `scripts/research/run_e136_case.sh`.
- Evaluation: `scripts/research/evaluate_e136_videos.py`.
- Large video/latent files:
  `/data1/models/svdquant-wjq/research/20261028/E136_h3_precision_windows/`;
  workspace preview copies: `results/media/E136/`.

Run one new arm with `bash scripts/research/run_e136_case.sh GPU CASE ARM`,
where `CASE` is `p016` or `vbench_000` and `ARM` is `ends_int8_g128` or
`ends_bf16_svd_sage3`. Existing outputs are not overwritten. Recompute a
case's paired metrics with
`CUDA_VISIBLE_DEVICES=GPU /home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python scripts/research/evaluate_e136_videos.py --case CASE`.

Limit: two deliberately selected prompts provide a mechanism screen, not
statistical evidence across prompts. The INT8 endpoint numerical proxy cannot
be used to infer native INT8 runtime or exact native-kernel rounding.
