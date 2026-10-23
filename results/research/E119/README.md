# E119 — split rare FP8 V tiles out of the Sage3 FP4 hot path

This is a performance/accuracy feasibility experiment, **not** an accepted
replacement for the native mixed-precision operator. Input is the same
MiniMax-H3 prompt-1, step-0, block-25 capture used in E118: noncausal
`[1,56,9482,128]` on RTX PRO 5000 72GB (GPU 0), with threshold 0.10
selecting 106/4,200 V tiles (2.52%). The corrected attention mask and
selected-only E117 V router are held fixed.

## Prototype

The ordinary FP4 core remains separate from all FP8 GEMMs. A private
`fp4attn_lse_cuda` build exposes each query row's quantized-QK softmax
log-sum-exp (LSE); its output is bitwise equal to the corrected pure FP4
core. This required writing the row sum already computed inside softmax.
On the full shape, LSE export changed pure-core CUDA-event time from
`4.726` to `4.765 ms` in its A/B test.

Three Triton stages then (1) compute selected tiles' approximate
`V8 − V4` after the existing sparse router, (2) compact route flags,
and (3) recompute only selected QK blocks using BF16 Q/K, weight the
V difference by `exp(score − LSE)`, and add it to the pure output.
No route is chosen or tuned from the benchmark result. The best of tested
query tiles (`BM=32,64,128`) was `BM=64`: sparse correction cost
`0.785 ms`; `BM=32` and `BM=128` cost `1.073` and `1.165 ms` respectively.
The separate V-delta and route-compaction stages cost `0.030` and
`0.013 ms`. The ordinary mixed core costs about `15.17 ms`; this split
core plus those stages costs about `5.48 ms`.

## Full online result and accuracy gate

Three independent processes each used 20 warmups and 8 batches × 12
CUDA-event calls. All three full online paths include preprocessing,
dynamic Q/K/V quantization, routing where applicable, attention core,
and output crop. They exclude the benchmark-only restoration of K, which
Sage3 preprocessing mutates.

| Online path | Mean time | Relative to pure FP4 |
| --- | ---: | ---: |
| Corrected pure FP4 | 7.826 ms | 1.000× |
| Original native mixed FP4/FP8 | 18.433 ms | 2.355× |
| Split sparse V correction | 8.851 ms | 1.131× |

The timing target is nearly met, but the **precision target is not**.
Against a full BF16 SDPA output on the same input:

| Output | NMSE vs BF16 | Improvement over pure FP4 |
| --- | ---: | ---: |
| Pure FP4 | 0.748434% | — |
| Native mixed | 0.745237% | 0.003197 percentage points |
| Split sparse V correction | 0.747540% | 0.000894 percentage points |

The fast variant retains only about **28%** of the original mixed path's
BF16-error improvement. Its full-output NMSE relative to native mixed is
`1.476e-5`, so it must not be labeled numerically equivalent to native
mixed. On a forced-all-FP8 128-token control, native mixed NMSE was
`0.324%` vs BF16, while V-only correction was `0.621%` (pure FP4
`0.855%`): the quality gap is much clearer when FP8 routing is dense.

An oracle scalar applied to the V correction cannot recover the missing
benefit: the least-squares scalar to BF16 is `0.715`, and it only improves
the full NMSE from `0.747540%` to `0.747370%`. A separate short-sequence
probe approximating both P8V8 and P4V4 from BF16-recomputed scores also
failed to match native mixed (`0.720%` vs BF16 instead of `0.324%`).
This indicates the missing term is structurally different, not merely a
mis-scaled V correction. The native path uses FP8 for the selected P tile
as well as V, while the split prototype retains pure FP4 P; furthermore,
recomputed BF16 QK scores are not identical to the native NVFP4 QK scores.

## Decision and reproduction

**Do not integrate this prototype into H3 generation.** A deployable
near-1.1× version must preserve the selected-tile P8 effect and use the
native quantized QK/softmax conventions. A promising next design is to
export the selected QK/P data from a spill-free FP4 pass and run rare FP8
PV work separately; its register, memory-traffic, and numerical costs are
unmeasured, so there is no claim that exact native-mixed accuracy at 1.1×
has been achieved.

Build LSE variant with `bash scripts/research/build_e119_sage3_lse.sh`.
Run `scripts/research/e119_validate_lse.py`,
`scripts/research/e119_sparse_v_correction.py --full-reference`, and
`scripts/research/e119_full_online_benchmark.py`. Process-level results
are under this directory. The LSE `.so` SHA-256 is
`09c3a8c1078aff8c01f8ecd1730bb3fdaebaac81fb654bbee3fc44b21bc8bbcf`.
Only one H3 capture/GPU has been tested; no cross-prompt or full-generation
quality claim is made.
