# E099 — H3 SVDQuant activation second-level scale, one full generation

Completed 2026-10-15 on physical GPU 6, sharing the card with an existing
~19 GiB job at the user's request. This is **not** a latency benchmark.

## Run contract

- Rank-64, grid-10 SVDQuant state:
  `results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt`
  (SHA-256 `d199da3e5c89d248653d72b035e57cc505644075f269c6f44cc4c552e1f10f73`).
- Prompt ID 1, seed 22529, restored from `/home/wjq/workspace/178866172854036`.
  This prompt is **in calibration/training**, not held-out.
- 480×832, 124 frames at 24 fps (~5.17 s), 20 denoising steps, CFG 1, tiled.
- The unmodified rank-64 SVDQuant `nvfp4_qdq` numerical path was used. An
  optional observer recorded only scalar statistics *after static smoothing*
  and *before activation fake quantization*; it did not change scale selection.
- Complete 4,000 calls = 200 target linears × 20 steps. The video decoded to
  124 H.264 frames at 832×480 and contains AAC audio. It is under
  `/data1/models/svdquant-wjq/research/20261015/E099_h3_svd_scale_p1/`.

## Results

The dynamic FP32 second-level scale is `amax(abs(x/s)) / (6×448)`, bounded
below by `1e-12`, for each complete linear input on each call. Here the
quantile population is **4,000 calls** (equal weight per call, not per element).

| Dynamic tensor-wise scale | p0 | p25 | p50 | p75 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All target calls | 0.000503 | 0.001192 | 0.002128 | 0.01814 | 0.2946 | 1.0893 | 3.8810 |

Call counts by scale interval: `<0.001`: 596; `[0.001,0.01)`: 2,237;
`[0.01,0.1)`: 676; `[0.1,0.5)`: 316; `[0.5,1)`: 127;
`[1,2)`: 21; `[2,4)`: 27.

| Projection (1,000 calls each) | scale p50 | scale p95 | scale max | elements with `abs(x/s)>2688` |
| --- | ---: | ---: | ---: | ---: |
| attention QKV | 0.001436 | 0.003464 | 0.004813 | 0 |
| attention output | 0.002999 | 0.020368 | 0.044829 | 0 |
| MLP FC1 | 0.001250 | 0.002407 | 0.003953 | 0 |
| MLP FC2 | 0.098214 | 0.976191 | 3.880952 | 48 |

Across **501,645,312,000** observed activation elements, just **48** exceed
2688 (fraction `9.57e-11`). Each affected call has one such element: 48/4,000
calls (1.2%). All 48 are in three FC2 layers: `blocks.39.mlp.fc2` (19 steps),
`blocks.45.mlp.fc2` (15), and `blocks.30.mlp.fc2` (14). The largest
`abs(x/s)` is **10,432** in block 45 FC2, step 18. The other 197 layers have
no value above 2688 on this run. The 48 calls are exactly the calls with
dynamic scale >1.

This says a fixed-second-level-scale-1 NVFP4 path would encounter **very rare
but potentially high-impact outliers** on this trajectory. It does not imply
the output-error impact is negligible: we did not rerun with Nunchaku's FP4
packing/GEMM, and changing the global scale also changes E4M3 block-scale
rounding below the 2688 threshold. The exact raw BF16 input is not preserved,
so this result cannot estimate clipping-induced output NMSE without another
instrumented comparison.

Numerical artifacts: `h3_svd_scale_p1_480p5s_calls.jsonl` and
`h3_svd_scale_p1_480p5s_summary.json`. Reproducer:
`scripts/research/launch_h3_svd_nvfp4_scale_audit.sh` and
`scripts/research/run_h3_svd_nvfp4_scale_audit.py`. Local run log:
`results/logs/e099_h3_svd_scale_audit.log`.

Video SHA-256: `1b5247fb2427ea3c5f00499115248166c3a38ce7406015f012248ebe9cf62202`.
