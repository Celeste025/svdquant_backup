# E094 — Video-only p99 SmoothQuant on H3 FC2

Completed 2026-10-14. This is an error-only, one-layer test of whether text/audio/video mixing makes static p99 SmoothQuant weak on real MiniMax-H3. It uses `blocks.25.mlp.fc2` (BF16 weight `[5376,14336]`), prompt 11 step 0 for calibration and prompt 1 step 0 for disjoint evaluation. Both measured inputs contain **only the 21,312 video-token rows**, and both errors use the original BF16 video-row `A @ W.T` as target. The 16 trailing padding rows in p1 are excluded. The FC2 linear is token-independent, so removing text/audio changes no BF16 video-row output; it can change p99 channel statistics and the per-tensor activation NVFP4 global scale.

The `mixed` calibration derives per-channel nearest-rank `p99(|A[:,j]|)` and static activation global scale from p11's full text/audio/video sequence. The `video_only` calibration derives both from p11 video rows only. Both use `s_j = p99_j^alpha / max|W[:,j]|^(1-alpha)`, transform `(A/s).to(BF16), (W*s).to(BF16)`, and use native FlashInfer K16 E2M1/E4M3 W4A4 packing and GEMM. Alpha grid `{0,.25,.5,.75,1}` is selected separately for each arm on **p11 video output NMSE**, never on p1. A current-input global-scale control recomputes only the p1 video activation scale; it leaves the p11-derived channel vector and quantized weight unchanged. Results are complete-linear output NMSE, not activation Q/DQ NMSE or model-quality scores.

| Calibration source | Alpha | p11 video, static | p1 video, static | p1 video, current-input scale |
|---|---:|---:|---:|---:|
| Mixed | none | 0.96466% | 1.01530% | 0.96824% |
| Video only | none | 0.96422% | 1.02773% | 0.96824% |
| Mixed | 0.25 | 0.81340% | 0.99063% | 0.92512% |
| Video only | 0.25 | 0.80914% | 0.99920% | 0.92426% |
| **Mixed** | **0.5, selected** | **0.73075%** | **0.96001%** | **0.95984%** |
| **Video only** | **0.5, selected** | **0.72761%** | **1.08629%** | **0.97224%** |

The video-only p99 recipe improves the p11 video calibration NMSE by only 0.43% relative to mixed calibration at selected alpha 0.5, but **worsens held-out p1 video NMSE by 13.15% relative under the static scale** (1.08629% versus 0.96001%). It also fails to improve on unsmoothed video-only W4A4 (1.02773%). Recomputing p1's activation global scale reduces the gap greatly, but video-only p99 is still 1.29% worse than mixed p99 (0.97224% versus 0.95984%) and 0.41% worse than the common unsmoothed control (0.96824%). Alpha 0.25 looks better on p1 with current-input scale, but it was *not* chosen on p11 and is post-hoc on this test.

The mixed and video-only p99 vectors correlate at 0.995. Their per-channel ratio has median 1.006; 1555/14336 channels (10.85%) differ by more than 10%. At alpha 0.5, excluding non-video rows reduces p11's smoothed activation amax from 844 to 388, while p1 video amax remains about 900–916. Thus the evaluation/calibration amax ratio rises from 1.09 to 2.32. This directly explains much of the static-scale failure: video-only calibration sets a tighter representable range that does not transfer to p1. The dynamic-scale control removes that particular mismatch but not all of the generalization gap. BF16 reparameterization roundoff is only ~8.6e-6 NMSE, far below the W4A4 error.

**Decision:** The tested one-call, modality-specific p99 smoothing is not supported as an improvement. The hypothesis that mixed-modality p99 calibration is the main cause of weak smoothing is not supported by this test; non-video rows actually act as a useful range guard here. This does not rule out per-modality quantization with separate scales, multi-prompt video calibration, or results in other layers/timesteps. No free-generation or whole-DiT quality claim is made.

Raw rows and calibration details: `h3_fc2_video_only_p99_nvfp4.json`. Reproducer: `scripts/research/eval_h3_video_only_p99_smoothing.py`. Log: `results/logs/e094_h3_video_only_p99_full.log`. The 64-row smoke file is a path check only and is not used for conclusions.
