# E146 — p025 retained 20% precision comparison

Same prompt/seed/initial latents and schedule as E145, except the video-region
route now skips 80% and retains 20%. Compare all retained work in INT8 against
an equal split of 10% total INT8 + 10% total NVFP4. Prefix Q/K/V use the same
INT8 path in both arms.

| arm | RGB NMSE vs BF16 | LPIPS-Alex mean | summed DiT time | vs BF16 time |
|---|---:|---:|---:|---:|
| 70% skip / 30% INT8 | **19.8716%** | **0.3216** | 176.32 s | -16.5% |
| 80% skip / 20% INT8 | 26.0894% | 0.3869 | **173.23 s** | **-18.0%** |
| 80% skip / 10% NVFP4 + 10% INT8 | 24.8975% | 0.3636 | 174.45 s | -17.4% |

Reducing retained work from 30% to 20% saves only another 3.09 s of summed DiT
time (about 1.8% relative to the 30%-INT8 arm) but increases closed-loop visual
distance substantially.  In this one rollout the 10%+10% mixed arm is closer to
BF16 than 20% pure INT8, despite including lower precision. This should not be
interpreted as a stable NVFP4 advantage: diffusion trajectories are nonlinear,
and the sampled frames show pose/motion divergence (for example a tongue-out
frame in the 20%-INT8 trajectory) dominating LPIPS. A multi-prompt comparison
is required for a precision-ranking claim.

All arms retain readable tile markings and the final “胡了！” text. The practical
single-case knee remains 70% skip / 30% INT8: it has the lowest BF16 distance,
while the additional 80% sparsity yields little end-to-end DiT time reduction.
