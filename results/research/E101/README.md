# E101 — MiniMax-H3 rank-32 SVDQuant activation tensor scale = 1

## Contract

- Original checkpoint: `results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt`, SHA-256 `e5ccf2d809f0b03e6a0eaef400e4f610ccfc37478eb3715d75040cf07446652d`. No recalibration or fine-tuning.
- Three inputs: H3 prompt 16 / seed 20026 (flying car, cliff, rainbow); VBench prompt 161 / seeds 38001610 and 38001611 (`A person is clapping`).
- All outputs: 576×1024, 124 frames, 20 denoising steps, CFG 1, video flow shift 12, audio flow shift 3, CPU seeded noise, tiled VAE, 24 FPS.
- SVDQuant: 200 target linears, group-16 NVFP4 W4A4, rank-32 branch, grid-10 smoothing. The three-case primary comparison uses the same BF16-GEMM fake-quantization runtime in both dynamic and fixed activation arms. Original packed-native E038 clapping videos are historical references, not the primary scale comparison.
- The fixed arm changes only the **activation tensor-wise scale** from `max(abs(x/s))/2688` to 1. The per-16-element local scale remains input-dependent and is capped at 448 before E4M3 storage. Weight, smoothing, and low-rank tensors are identical between arms.
- Generation used GPU 4/5/6 in separate named tmux sessions, with a 45,000 MiB free-VRAM guard before each run. No pre-existing GPU process was stopped.

## Results

The primary metric is mean of 20 teacher-forced, same-input, full-DiT per-step NMSEs. Positive relative change means fixed scale has **more** error. These are three selected cases, not a population significance test.

| Case | Video NMSE dynamic → fixed | Relative change | Steps fixed lower | Audio NMSE dynamic → fixed | Relative change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Race / 20026 | 7.417% → 7.487% | +0.93% | 6/20 | 3.866% → 3.742% | −3.22% |
| Clap / 38001610 | 2.263% → 2.345% | +3.59% | 4/20 | 1.694% → 1.682% | −0.70% |
| Clap / 38001611 | 2.940% → 3.038% | +3.30% | 3/20 | 2.312% → 2.537% | +9.75% |

The decoded 8-bit H.264 RGB NMSE against separately generated BF16 videos is 51.91% → 50.44% (race), 57.20% → 61.26% (clap0), and 36.24% → 38.74% (clap1). This is a **free-rollout visual-distance proxy**, not a DiT quantization NMSE or perceptual quality score: small local changes can make trajectories diverge in pose, framing, and lighting. In particular, the race decoded-video distance and its teacher-forced DiT NMSE move in opposite directions.

## Artifacts

- `summary.json`: validated three-case numerical summary.
- `{race,clap0,clap1}/svdquant{,_fixed1}.{json,csv}`: every paired DiT call's video/audio NMSE, MSE, RMSE, cosine, max-absolute error, and timestep.
- `{race,clap0,clap1}/video_pixel_comparison.json`: pooled and per-frame decoded RGB MAE/MSE/NMSE/PSNR.
- Videos, four-frame posters, and BF16 teacher-call caches: `/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases/`.
- Three-column montage order is BF16 original, dynamic SVDQuant, fixed-scale SVDQuant. Each montage has 124 frames at 24 FPS and no audio; individual generated MP4s retain audio.
- Generation and replay logs: `results/logs/e101_h3_{race,clap0,clap1}{,_error}.log`.
- Source: `scripts/infer_minimax_h3_svdquant_standard.py`, `scripts/minimax_h3_svdquant_common.py`, `scripts/eval_minimax_h3_paired_step_error.py`, and `scripts/research/{launch_h3_fixed1_three_cases.sh,launch_h3_fixed1_three_cases_error.sh,analyze_h3_fixed1_three_cases.py,summarize_h3_fixed1_three_cases.py}`.

## Interpretation

For these three cases, fixing the activation global scale to 1 does **not** improve full-DiT video-output fidelity; the mean video NMSE rises by about 0.9–3.6% relative to the original dynamic-scale checkpoint. Audio is mixed. The checkpoint was calibrated with dynamic activation scales, so this does not answer whether a separately recalibrated fixed-scale checkpoint could perform better. The fixed-scale arm remains a fake-quantization approximation, not a bit-exact Nunchaku run.
