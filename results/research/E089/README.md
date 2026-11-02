# E089 — H3 FC2 NVFP4 second-level scale granularity (error only)

Completed 2026-10-14. This is an exploratory **single-linear, two-input** result,
not whole-DiT quality or speed. The target is MiniMax-H3
`blocks.25.mlp.fc2`: A is `[tokens, 14336]`, W is `[5376, 14336]`, and the
output is `A @ W.T`. Native FlashInfer NVFP4 always packs 16 contiguous values
along K. Only the second-level FP32 global scale changes granularity. Here an
"activation row" means one token; a "weight row" means one output channel.

The calibrated-tensor baseline uses an activation scale from the *other*
cached input and one per-tensor weight scale. An additional current-input
per-tensor scale isolates the benefit of input adaptation from the benefit of
row grouping. `p1` uses `p11` for its calibration scale; `p11` uses `p1`.
Neither pair was used to tune these group sizes. All group scales are native
absmax-derived scales, **not** NMSE-optimized scales.

## Output NMSE

Numbers are NMSE × 100 (percent of BF16 output energy). A-only and W-only
keep the other operand at the calibrated-tensor/per-tensor baseline; "both"
groups both operands at the stated row count.

| Input | Scale choice | A-only | W-only | Both |
|---|---|---:|---:|---:|
| p1, step 0 | Cross-input calibrated tensor | 1.042448% | 1.042448% | 1.042448% |
| p1, step 0 | Current-input tensor A, tensor W | 0.964316% | — | — |
| p1, step 0 | 512 rows | 0.964448% | 1.042481% | 0.964923% |
| p1, step 0 | 128 rows | 0.963703% | 1.043194% | 0.964091% |
| p1, step 0 | 32 rows | 0.962690% | 1.042152% | 0.962311% |
| p1, step 0 | 1 row | **0.957224%** | 1.041726% | **0.956693%** |
| p11, step 0 | Cross-input calibrated tensor | 0.960359% | 0.960359% | 0.960359% |
| p11, step 0 | Current-input tensor A, tensor W | 0.960738% | — | — |
| p11, step 0 | 512 rows | 0.960502% | 0.959615% | 0.959804% |
| p11, step 0 | 128 rows | 0.960050% | 0.960575% | 0.960306% |
| p11, step 0 | 32 rows | 0.959138% | 0.961242% | 0.960016% |
| p11, step 0 | 1 row | **0.953112%** | 0.958581% | **0.951287%** |

For p1, current-input per-tensor A reduces NMSE by **7.50% relative** to the
cross-input baseline. Refining that current-input scale to per-row reduces
NMSE by only another **0.735% relative**. For p11, current-input per-tensor
is essentially tied with its cross-input baseline (-0.040% relative benefit),
but per-row A improves **0.794% relative** to current-input per-tensor.
The 32-row A case improves only 0.169% and 0.167%, respectively; 128/512
rows are smaller still. Weight-only per-row changes the cross-input baseline
by +0.069% and +0.185% relative benefit on p1/p11. Grouping both operands
per-row improves current-input per-tensor by 0.790% and 0.984% relative.

## Numerical validity and interpretation

FlashInfer 0.7.0.post1's native quantizer produces the actual packed E2M1
codes and E4M3 K-block-16 scales in every case. The installed dense `mm_fp4`
supports only a scalar second-level `alpha`, so grouped cases decode these
native codes/scales and use FP32 GEMM followed by BF16 output rounding.
The per-tensor decoded-GEMM baseline matches the native FP4 Tensor-Core GEMM
with output-to-output NMSE `3.55e-9` (p1) and `3.42e-9` (p11), below the
predeclared `2e-5` gate. Native baseline output NMSEs match the decoded
baseline to displayed precision. Thus the proxy is numerically faithful for
this input, while it makes **no claim about grouped-kernel deployability or
speed**.

Decision: these two inputs do not justify implementing a custom grouped-scale
dense GEMM yet. The largest observed gain attributable to row granularity
after a current-input tensor scale is <1% relative NMSE, versus the ~1%
baseline error. The more consequential question for p1 is whether the fixed
cross-input activation scale is appropriate. This result does not rule out
larger gains in other layers, timesteps, or with NMSE-optimized scales.

Raw results: `h3_fc2_p1_s00_row_scales.json` and
`h3_fc2_p11_s00_row_scales.json`. Reproducer:
`scripts/research/eval_h3_nvfp4_row_scales.py`. Local logs:
`results/logs/e089_h3_row_scale_p1.log` and
`results/logs/e089_h3_row_scale_p11.log`.
