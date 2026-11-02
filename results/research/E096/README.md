# E096 — Removing the top 0.1% of H3 FC2 video activations

Completed 2026-10-14. Real MiniMax-H3 `blocks.25.mlp.fc2` weight and two cached step-0 calls are used: p11 video (calibration) and p1 video (independent evaluation), each `[21312,14336]`. Within **each raw video activation tensor**, the exact largest `ceil(0.001 × 305,528,832) = 305,529` entries by absolute value are set to zero. BF16 ties at the cutoff are resolved by lowest flattened index. The cutoff is 81.0 for p11 and 84.5 for p1. Text, audio and p1 padding rows are not included. The two W4A4 arms use the same cleaned input and the same **cleaned BF16 `X_clean W.T` output** as reference.

Plain is native FlashInfer K16 E2M1/E4M3 W4A4. SmoothQuant-style uses `s_j = p99_calibration|X_j|^0.5 / max|W[:,j]|^0.5` and BF16-cast `X/s, W*s` before native W4A4. The smoothing vector and static activation global scale are **recalibrated on cleaned p11** for the cleaned arm. Alpha 0.5 was fixed from the preceding E094 calibration choice, not tuned on cleaned p1. Dynamic-control columns recompute only the current activation's global scale. No SVDQuant low-rank branch is present.

## Cleaned-BF16-target output NMSE

| Data / call | Plain W4A4, static / current scale | P99 smoothing + W4A4, static / current scale |
|---|---:|---:|
| No cleanup, p11 calibration | 0.96422% / 0.96422% | 0.72761% / 0.72761% |
| No cleanup, p1 held-out | 1.02773% / **0.96824%** | 1.08629% / **0.97224%** |
| Top 0.1% zeroed, p11 calibration | 1.14594% / 1.14594% | 0.97906% / 0.97906% |
| Top 0.1% zeroed, p1 held-out | 1.14264% / **1.14462%** | 1.12040% / **1.11975%** |

With the current-input scale, p1 **plain** NMSE rises from 0.96824% to 1.14462% (+18.22% relative) after cleanup; **smoothed** rises from 0.97224% to 1.11975% (+15.17%). Smoothing beats plain on the cleaned p1 target by 0.02487 percentage points (2.17% relative), but that is a modest gain and does not reverse the fact that cleanup made both quantized computations *less* accurate relative to their respective BF16 targets. The larger p11 in-sample smoothing gain (1.14594% to 0.97906%, 14.56% relative) transfers only weakly to p1.

The NMSE increase is a **denominator effect, not an increase in absolute squared error**. On p1, cleanup leaves only 59.10% of the original BF16 output energy. Plain W4A4 squared-error energy falls to 69.86% of its original value (−30.14%), but signal energy falls more (−40.90%); therefore `NMSE_clean / NMSE_original = 0.6986 / 0.5910 = 1.1822`. For smoothed W4A4, absolute squared-error energy falls to 68.06% (−31.94%), while the same signal-energy denominator falls to 59.10%, producing a 15.17% relative NMSE rise. Removing tail values **does** reduce absolute error; it removes even more of the output energy used to normalize that error.

The cleaned BF16 function is radically different from the original one: `NMSE(X_clean W.T, XW.T)` is 36.23% for p11 and 35.16% for p1; the cleaned output energy is only 58.30% and 59.10% of the original, respectively. These figures are **not** included in the cleaned-target W4A4 NMSE above; they matter if one proposes deploying the zeroing without preserving the discarded values. The result therefore argues against literal top-0.1% removal, while leaving a high-precision sparse outlier bypass as a separate untested method.

Interpretation: In this layer/prompt pair, the remaining ~1% W4A4 output NMSE is not simply caused by the most extreme 0.1% of raw activation elements. Static global-scale mismatch is also much smaller after cleanup (p11/p1 raw amax 81/84.5). Smoothing still has a measurable but small **held-out** relative gain under the altered computation. This does not establish that smoothing is useless generally, or that an optimized cleaned-data alpha would give the same result. It is one layer, one timestep and two prompts, with no formal significance claim.

Raw results: `h3_fc2_top01pct_clean_video_nvfp4.json`. Reproducer: `scripts/research/eval_h3_top01pct_clean_nvfp4.py`. Log: `results/logs/e096_h3_top01pct_clean_nvfp4.log`. No-clean native baselines reproduce E094 within `2e-7` absolute NMSE; exact mask counts, finite outputs and shared-target comparisons passed schema checks.
