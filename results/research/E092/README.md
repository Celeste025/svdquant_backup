# E092 — H3 FC2 signed INT4 W4A4 smoothing, K-group 16 vs 128

Completed 2026-10-14. Error-only, two-call diagnostic on real `blocks.25.mlp.fc2`: p11 step 0 calibrates the per-input-channel smoothing vector and selects alpha; separate p1 step 0 is held out. The full `[22464,14336]` and `[22400,14336]` activations and original BF16 `[5376,14336]` weight are used. Inputs/calls/weight are identical to E090/E091. The alpha grid is `0, 0.25, 0.5, 0.75, 1`, plus an unsmoothed arm.

Both activation and weight are signed `[-8,7]` INT4, round-to-nearest. For each contiguous K-group, the symmetric zero-point-free FP32 scale is `max(max(x)/7, -min(x)/8)`. Activation scales are recomputed for each row/token and K-group, independently for p11 and p1. Weight scales are per-output-row and K-group and computed offline from the smoothed weight. Quantized codes are decoded to BF16 and multiplied with BF16 GEMM; the BF16 original `A @ W.T` is the NMSE reference. This is **not** a native INT4 W4A4 deployment kernel or speed benchmark. On eight real rows the decoded-BF16 GEMM versus groupwise integer-code product gate is `4.03e-6` (K16) and `4.98e-6` (K128) NMSE; the gate does not certify a full native kernel.

## Calibration-selected results

NMSE values are percentages of original BF16 output energy; lower is better. The alpha in each smoothing arm is chosen **only on p11 calibration NMSE**.

| INT4 K-group | Smoothing statistic | Selected alpha | p11 NMSE | p1 NMSE | p1 change vs plain |
|---:|---|---:|---:|---:|---:|
| 16 | none | — | 0.84871% | 0.86118% | — |
| 16 | absmax | 0.25 | 0.74307% | 0.84902% | 1.41% lower |
| 16 | abs-p99 | 0.5 | **0.66883%** | 1.11442% | **29.41% higher** |
| 128 | none | — | 2.58534% | 2.60357% | — |
| 128 | absmax | 0.25 | 2.26881% | **2.59817%** | 0.21% lower |
| 128 | abs-p99 | 0.25 | **2.17133%** | 2.68005% | **2.94% higher** |

K128 increases plain INT4 p1 NMSE from 0.861% to 2.604%, so coarser K grouping does make this particular quantizer substantially less accurate. Nevertheless, p99 smoothing improves p11 calibration and fails to improve p1 with its calibration-selected alpha under **both** group sizes. Even the per-token, per-group activation scale is dynamic on p1, so this reversal is not the fixed global-scale failure seen in E090. Absmax smoothing has only a small p1 gain. This is evidence against the narrow claim that K128 INT4 alone makes a one-call p99 smoothing recipe valuable for held-out H3 FC2 inputs.

This is not an isolated NVFP4-versus-INT4 granularity comparison: the codebook, scale representation and native execution differ. The matched INT4 K16/K128 arms isolate K-group granularity within the specified INT4 quantizer. Only two prompt/seed calls and one linear are tested, so no broad model-quality or generalization claim follows.

Raw full result: `h3_fc2_p11cal_p1eval_int4_g16_g128_smoothing.json`. Reproducer: `scripts/research/eval_h3_int4_group_smoothing_fc2.py` with the recorded inputs/GPU. Log: `results/logs/e092_int4_g16_g128_full.log`. The 128-row smoke file is explicitly **not** a reportable full-call result.
