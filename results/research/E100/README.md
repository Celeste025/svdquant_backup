# E100 — H3 rank-64 SVDQuant with fixed activation second-level scale

## Contract

- Prompt ID 1, seed 22529 from `/home/wjq/workspace/178866172854036` (in the original calibration set).
- 480×832, 124 frames, 20 denoising steps, CFG 1, tiled, 24 FPS.
- Rank-64 standard SVDQuant state: `results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt`, SHA-256 `d199da3e5c89d248653d72b035e57cc505644075f269c6f44cc4c552e1f10f73`.
- BF16 original and fixed-scale SVDQuant generated sequentially on shared physical GPU 5, each only after a 45,000 MiB free-VRAM guard. This is not a speed benchmark.
- Dynamic-scale SVDQuant video reused from E099: `/data1/models/svdquant-wjq/research/20261015/E099_h3_svd_scale_p1/svdquant_480x832_124f_20steps.mp4`.

## Fixed-scale simulation

Only activation quantization changes. For each 16-element group of the statically smoothed input `x/s`, calculate `raw = clamp(max(abs(group))/6, 1e-12, 448)`, store the group scale as E4M3, and round the normalized activation to E2M1. The global activation tensor-wise scale is fixed to 1. The low-rank branch, smoothing, and quantized main weights are unchanged. This uses the unrounded capped local scale for E2M1 rounding and the E4M3-rounded scale for Q/DQ reconstruction, matching the relevant Nunchaku kernel ordering. It remains a BF16-GEMM fake-quantization approximation, not a bit-identical packed Nunchaku inference path.

The original PTQ checkpoint was calibrated with dynamic activation tensor-wise scales. This experiment evaluates the *existing checkpoint* under fixed-scale activation quantization; it does not estimate the best achievable quality after recalibration.

## Artifacts and comparisons

- New videos and BF16 teacher-call cache: `/data1/models/svdquant-wjq/research/20261015/E100_h3_svd_fixed1_p1/`.
- Generation source: `scripts/infer_minimax_h3_svdquant_standard.py`, `scripts/minimax_h3_svdquant_common.py`; launcher: `scripts/research/launch_h3_svd_fixed1_p1.sh`.
- Paired teacher-forced DiT-output video/audio NMSE, MSE, RMSE, max-absolute-error, and cosine: `svdquant_p1_480x832_124f_20steps.{json,csv}` and `svdquant_fixed1_p1_480x832_124f_20steps.{json,csv}`. Both use the exact same cached BF16 DiT inputs and outputs; launcher: `scripts/research/launch_h3_fixed1_paired_error.sh`.
- Decoded H.264 8-bit RGB frame comparison against separately generated BF16 video: `video_pixel_comparison.json`. This is a free-generation visual-distance proxy, **not** DiT-output error. Script: `scripts/research/analyze_h3_fixed1_videos.py`.
- Logs: `results/logs/e100_h3_fixed1_p1.log` and `results/logs/e100_h3_fixed1_error.log`.

## Results

All three videos decode to 832×480, 124 frames, with one audio stream. The paired numerical reports contain all 20 steps, finite video/audio metrics, and identical BF16 timestep inputs for the two quantized arms.

| Teacher-forced full-DiT output | Dynamic scale | Fixed scale 1 | Fixed / dynamic |
| --- | ---: | ---: | ---: |
| Video: mean of 20 per-step NMSEs | 4.4961% | 4.6161% | 1.0267× |
| Video: pooled energy-normalized MSE | 4.4445% | 4.5825% | 1.0311× |
| Audio: mean of 20 per-step NMSEs | 2.7544% | 2.9638% | 1.0760× |
| Audio: pooled energy-normalized MSE | 2.9009% | 3.1427% | 1.0833× |

Fixed-scale NMSE is lower on 7/20 video steps and 5/20 audio steps, higher on the remaining 13/20 and 15/20. Video step 0 changes from 2.5340% to 2.7454% (+8.34% relative); the largest video relative degradation is step 6 (+16.40%), and the largest improvement is step 2 (−7.77%). Audio step 0 improves from 11.3041% to 6.8950%, but this does not reverse the 20-step mean increase. These are descriptive results for **one in-sample prompt**, not a significance or generalization claim.

Decoded-video RGB MSE against the separately generated BF16 MP4 is 1,681.1 for dynamic scale and 1,738.5 for fixed scale (+3.41%); PSNR is 15.875 versus 15.729 dB. This video-space result agrees in direction with the paired DiT video NMSE, but incorporates trajectory divergence and lossy H.264 encoding. It must not be interpreted as quantization-layer NMSE.

The result suggests that switching the existing dynamic-scale-calibrated rank-64 checkpoint to a fixed activation second-level scale of 1 causes a **small video-output error increase and a larger audio-output error increase** for this case. It does not isolate which layers or E4M3-rounding changes cause that difference, and it is not evidence that recalibrated fixed-scale SVDQuant would perform similarly.
