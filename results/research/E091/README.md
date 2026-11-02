# E091 — H3 FC2 SmoothQuant: activation abs-p99 vs absmax

One real `blocks.25.mlp.fc2` linear, p11 step 0 calibration and disjoint p1 step 0 evaluation. This repeats E090 with only the per-input-channel activation statistic changed from `max |A[:,j]|` to the nearest-rank empirical p99 of `|A[:,j]|` over all calibration rows. The weight statistic remains per-input-channel absmax. Alpha grid, native FlashInfer NVFP4 W4A4 packing/GEMM, BF16 reference, and alpha selection on p11 are unchanged. Full tensors were used, including packed-sequence padding. No SVDQuant low-rank branch, block endpoint, runtime profile, or generation test is included.

The table gives complete-linear output NMSE in percent of BF16 output energy. “p1 static” keeps the p11-derived activation global NVFP4 scale; “p1 dynamic” recalculates only that global quantization scale from the current p1 input while keeping the p11-derived smoothing vector and weight fixed.

| Activation statistic | alpha | p11 calibration | p1 static | p1 dynamic | p1/p11 smoothed amax |
|---|---:|---:|---:|---:|---:|
| none | — | 0.96074% | 1.04245% | 0.96432% | 2.05x |
| absmax | 0.25 | 0.85745% | 1.19887% | 0.93104% | 4.34x |
| abs-p99 | 0.25 | 0.81576% | 1.07370% | 0.92764% | 4.57x |
| absmax | 0.5 | 0.80352% | 1.61375% | 0.93261% | 10.00x |
| **abs-p99** | **0.5** | **0.73950%** | **1.05305%** | **0.97930%** | **4.64x** |

Alpha 0.5 is selected on p11 under both statistics. Replacing absmax with abs-p99 improves p11 calibration NMSE and substantially reduces the p1 static-scale failure, but the selected p99 recipe still **does not beat no smoothing on p1**: 1.05305% versus 1.04245% with static global quantization scale; 0.97930% versus 0.96432% with current-input global scale. P99 alpha 0.25 reaches 0.92764% under dynamic global scale, slightly better than absmax alpha 0.25, but it is a post-hoc choice on p1 and cannot be presented as selected by calibration. These two calls cannot establish generalization.

Raw result: `h3_fc2_p11cal_p1eval_absp99_nvfp4.json`. Reproducer: `scripts/research/eval_h3_smoothquant_nvfp4_fc2.py --activation-stat abs-p99` with the calibration/evaluation paths and GPU recorded in the JSON. Log: `results/logs/e091_h3_fc2_absp99.log`. E090 absmax reference: `results/research/E090/h3_fc2_p11cal_p1eval_smoothquant_nvfp4_diagnostic.json`.
