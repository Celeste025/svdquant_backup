# E102 — H3 FC1 activation global-scale paired comparison

On the real BF16 teacher-forced prompt 1, step 0 call (480×832, 124 frames), capture the complete input of `blocks.0.mlp.fc1` and its BF16 weight. Compare the existing rank-32 SVDQuant activation quantization recipes with the same input, weight, smoothing, low-rank A/B, and quantized residual weight. The BF16 `Wx` is the common reference. The full 456 MiB input/weight capture is local-only on `/data1` and is identified in the JSON by SHA256. Rank-32 checkpoint SHA256: `e5ccf2d809f0b03e6a0eaef400e4f610ccfc37478eb3715d75040cf07446652d`.

| Metric | Dynamic global scale | Fixed global scale = 1 | Fixed minus dynamic |
|---|---:|---:|---:|
| SVDQuant linear output NMSE | 1.2368% | 1.2600% | +0.0232 percentage points (+1.88% relative) |
| Main-branch output NMSE | 1.2358% | 1.2593% | +0.0234 percentage points (+1.90% relative) |
| Activation QDQ NMSE | 0.8431% | 0.8770% | +0.0340 percentage points (+4.03% relative) |

The measured dynamic FP32 tensor scale is `0.0007178897`, not an assumed value. Of 5,225,472 local 16-element scales, 7.69% are below the E4M3 minimum *normal* value when the global scale is 1, but only 6 are below its minimum *subnormal* value. Most local scales therefore still fit in E4M3, explaining why the small global scale does not dramatically improve error in this case. Both arms continue to compute the local scale from the current activation.

This is a one-layer, one-call test, not evidence that fixed-one is uniformly safe over the model or denoising trajectory. The two arms use the repository's existing runtime recipes; besides the FP32 global scale, they differ slightly in FP4 code rounding: the fixed-one path divides by the raw local scale before its E4M3 storage rounding, while the dynamic path divides by the rounded effective local scale. Thus the reported delta is the effect of *those deployed-simulation recipes*, not a pure isolated mathematical intervention on one scalar.

Data: [`p1_step0_b0_fc1.json`](p1_step0_b0_fc1.json). Reproduction: `scripts/research/launch_e102_h3_fc_scale.sh`.
