# E095 — Zeroing activation outliers before H3 native NVFP4 W4A4

Completed 2026-10-14. This is a literal **zeroing** experiment, not clipping, a high-precision outlier bypass, or SVDQuant. Real `blocks.25.mlp.fc2` BF16 weight and p1/p11 step-0 cached calls are used. A modified activation `X_0` zeros entries with `|X| > T` before native FlashInfer K16 E2M1/E4M3 W4A4 packing and GEMM. The primary output error always compares the resulting W4A4 output against the **original, unmodified** BF16 `XW.T`. All reported quantities are complete-linear output NMSE; the denominator in the table below is the original BF16 output energy, except where explicitly marked as modified-target NMSE.

Three errors distinguish the mechanism:

- **Final:** `||Q(X_0)Q(W)^T - XW^T||² / ||XW^T||²` (the metric that matters if outliers are discarded).
- **Removal:** `||X_0W^T - XW^T||² / ||XW^T||²` (BF16 signal destroyed by zeroing).
- **Quantization-only:** `||Q(X_0)Q(W)^T - X_0W^T||² / ||XW^T||²`. A second version divides by `||X_0W^T||²` to show whether the modified problem itself became relatively easier.

The weight packet is identical in all arms. “Current-input scale” recomputes the FP32 NVFP4 activation global scale after zeroing; “fixed scale” keeps the original p11 scale, amax 2512. The no-zero controls reproduce E089 native output NMSE within `2e-7` absolute.

## Plain W4A4 results

| Call | Zero entries with `|X| >` | Removed / total | New activation amax | Quantization-only, current scale / original denominator | Removal NMSE | Final NMSE, current scale | Final NMSE, fixed p11 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1 | none | 0 / 321.1M | 5152 | 0.9643% | 0 | **0.9643%** | **1.0424%** |
| p1 | 2512 | 9 / 321.1M | 2272 | 0.9597% | 0.5632% | **1.5242%** | **1.5247%** |
| p1 | 1024 | 186 / 321.1M | 1024 | 0.9485% | 2.1858% | **3.1371%** | **3.1386%** |
| p1 | 512 | 1983 / 321.1M | 512 | 0.9151% | 6.7402% | **7.6683%** | **7.6697%** |
| p1 | 256 | 13737 / 321.1M | 256 | 0.8592% | 13.9399% | **14.8258%** | **14.8279%** |
| p11 | none | 0 / 322.0M | 2512 | 0.9607% | 0 | **0.9607%** | **0.9607%** |
| p11 | 2512 | 0 / 322.0M | 2512 | 0.9607% | 0 | **0.9607%** | **0.9607%** |
| p11 | 1024 | 101 / 322.0M | 1024 | 0.9531% | 1.0792% | **2.0322%** | **2.0325%** |
| p11 | 512 | 1460 / 322.0M | 512 | 0.9257% | 4.9683% | **5.9001%** | **5.9008%** |
| p11 | 256 | 12449 / 322.0M | 256 | 0.8678% | 12.5610% | **13.4515%** | **13.4525%** |

For p1 the nine entries over 2512 are one text and eight video values; no audio or padding values are removed. They lower the current-input amax by 2.27× when zeroed, but the final original-target NMSE rises from 0.9643% to 1.5242%, a **58% relative increase**. The quantization-only numerator relative to the original output energy falls only from 0.9643% to 0.9597%. Against the *modified* BF16 output, quantization-only NMSE actually rises from 0.9643% to 0.9663%; the apparently lower original-denominator figure partly reflects removed output energy. Under the fixed p11 scale, zeroing cuts quantization-only original-denominator NMSE from 1.0424% to 0.9606%, showing some range-mismatch benefit, but the lost signal overwhelms it. More aggressive zeroing worsens the final error sharply on both calls.

## Targeted E094 p99-smoothing check

We also reproduced E094's p11-video-only p99 SmoothQuant at calibration-selected alpha 0.5 and evaluated p1's 21,312 actual video rows. The p11 smoothed activation amax is 388, while p1 video amax is 900. Exactly 91 p1 video entries exceed 388; zeroing them brings the amax down to 388.

| p1 video, p99 alpha 0.5 | Removed | Quantization-only / original denominator | Removal NMSE | Final original-target NMSE |
|---|---:|---:|---:|---:|
| No zeroing, fixed p11 scale | 0 | 1.0863% | 0 | **1.0863%** |
| Zero values above 388, fixed p11 scale | 91 | 0.9540% | 0.7491% | **1.7027%** |

The no-zero fixed and current-scale outputs exactly reproduce E094 (1.08629% and 0.97224%). Literal zeroing reduces the quantization component, but increases final error by **56.7% relative** because those 91 values have a non-negligible contribution to the original output. This suggests a *preserved* sparse outlier path could be worth a separate experiment, but no such path is implemented or measured here.

## Same cleaned video tensor and same cleaned BF16 target

For a strictly paired plain-versus-smoothing comparison, `h3_fc2_matched_clean_video_nvfp4.json` evaluates **both** arms on p1's same 21,312 video rows, applies the **same element mask** before either arm, and uses that exact cleaned tensor's BF16 `X_clean W.T` as the shared target. The p11-video calibration and p99 alpha 0.5 remain fixed. Two pre-existing outlier definitions are shown: raw values above p11 video's raw amax (2336), or values that would exceed p11 video's smoothed amax (388). Static scales remain arm-specific p11 calibration scales; current-input scales are recomputed from the same cleaned input after each arm's representation transform.

| Shared p1-video mask | Removed | Plain W4A4, static / current scale | P99 SmoothQuant + W4A4, static / current scale |
|---|---:|---:|---:|
| None | 0 | 1.02773% / 0.96824% | 1.08629% / 0.97224% |
| Raw values > 2336 | 8 | 0.97015% / 0.97045% | 0.99224% / 0.97309% |
| Smoothed values > 388 | 91 | **0.97137% / 0.97260%** | **0.96313% / 0.96313%** |

Under the 91-entry mask, smoothing beats plain by only 0.00824 percentage points with static scales, or 0.00947 points (~0.97% relative) with current-input scales. Under the 8-entry raw-amax mask, smoothing is worse. Thus **once the cleaned multiplication is redefined as correct**, the answer is approximately 0.97% NMSE for plain NVFP4 and 0.96% for p99-smoothed NVFP4 under the 91-entry mask, not a large gap. The earlier 0.96273% smoothing figure used `X_clean/s @ (W*s).T` as its modified BF16 target; the paired table uses the common unsmoothed `X_clean W.T` target, accounting for a small BF16 reparameterization-roundoff difference. The masked target is intentionally a different function from the original H3 linear.

**Decision:** Extreme values do contribute to static-scale mismatch, especially after p99 smoothing, but they are not expendable. Literal zeroing is not a useful fix for this layer/prompt pair; it turns a quantization-range issue into a larger missing-signal error. The result does not prove what dominates across all H3 layers, prompts, or timesteps, and it is not a formal significance test.

Raw data: `h3_fc2_outlier_zeroing_nvfp4.json`, `h3_fc2_smoothed_outlier_zeroing_nvfp4.json`, and `h3_fc2_matched_clean_video_nvfp4.json`. Reproducers: `scripts/research/eval_h3_outlier_zeroing_nvfp4.py`, `scripts/research/eval_h3_smoothed_outlier_zeroing_nvfp4.py`, and `scripts/research/eval_h3_matched_clean_nvfp4.py`. Logs: `results/logs/e095_h3_outlier_zeroing_nvfp4_qacheck.log`, `results/logs/e095_h3_smoothed_outlier_zeroing_nvfp4.log`, and `results/logs/e095_h3_matched_clean_video_nvfp4.log`.
