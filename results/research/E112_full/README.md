# E112: H3 block-25 mixed P/V precision, complete attention output

This is a PyTorch numerical reference, **not** a native FP4/FP8 kernel speed
benchmark. Input is the real 480p/3s H3 prompt-1 step-0 block-25 QKV capture
identified and SHA-256-hashed in `summary.json`.

## Scope and numerical method

The main attention sequence has 9,482 tokens and 56 heads of dimension 128;
the second independent sequence has 54 tokens. All query rows of both sequences
were evaluated. The main sequence is padded to 9,600 keys for 75 KV tiles of
128 tokens. Each `(head, KV tile)` uses V's FP4 quantize/dequantize relative-L2
error to choose FP8 when error is at least 0.10. The same routing decision is
used for P and V in every query-block × KV-tile PV operation.

BF16-reference output is unquantized PyTorch SDPA on the captured QKV. The
quantized arms share Sage3-style centered NVFP4 Q/K numerics and the same
online-softmax denominator. The P/V variants are all-FP4, all-FP8, threshold
0.10 mixed, and top-1%-V-error mixed. FP8 is E4M3FN with dynamic max/448
per 16 elements along K for both P and V, matching MixAttention's **Python
prototype**, not its different CUDA experimental P scaling. QDQ values and
GEMMs are float32; this is not a claim about native packed-kernel timing.

## Complete-output results

NMSE is `sum((estimate-reference)^2)/sum(reference^2)`. The first column is
over the complete two-sequence 9,536 × 56 × 128 output tensor; the second
shows the 9,482-token main sequence alone.

| Arm | Complete output NMSE vs BF16 | Main sequence alone | Main change from all-FP4 |
|---|---:|---:|---:|
| Q/K NVFP4 only; unquantized P/V | 0.52484% | 0.52751% | diagnostic lower bound |
| P/V all FP4 | 0.75456% | 0.75804% | baseline |
| P/V all FP8 | 0.53306% | 0.53578% | 29.32% lower |
| P/V mixed, V-error threshold 0.10 | 0.75157% | 0.75504% | 0.397% lower |
| P/V mixed, top 1% V-error tiles | 0.75446% | 0.75795% | 0.013% lower |

Threshold 0.10 selects 106/4,200 `(head, KV tile)` pairs (2.5238%). With 75
query blocks per head, that is 7,950/315,000 main-sequence PV tile GEMMs, also
2.5238%; the short sequence selects 0/56 pairs and 0/56 GEMMs. Of 106 selected
main tiles, 55 are KV tile 6 and 29 are KV tile 5 (zero-based): routing is
concentrated in adjacent early-token regions, but upgrading them hardly changes
the final attention output NMSE.

For the 54-token short sequence, all-FP4 and threshold-mixed are identical at
0.07105% NMSE; all-FP8 is effectively exact against its unquantized reference.
The short sequence has very different length/energy and should not be pooled
unweighted with the main result.

The main V FP4 relative-L2 error is narrowly distributed: median 0.09628,
95th percentile 0.09895, 99th percentile 0.10324. Thus the 0.10 threshold is
sensitive to small threshold changes: at 0.08 all tiles route to FP8; at 0.12
none do. `summary.json` contains the complete threshold sweep and raw metrics;
`tile_errors.csv` contains every head/tile error and selection; the PNG shows
the error heatmap and both routing masks.

## Interpretation and limits

V-only error is a poor selector of tiles that matter most to final attention
output in this one case: using FP8 on 2.52% of V tiles captures only 1.35% of
the total all-FP4-to-all-FP8 output-NMSE improvement. This does **not** prove
that mixed precision is unhelpful; a P/PV-output-impact criterion, another
threshold, or other H3 blocks/steps/prompts may behave differently. No
generation-quality or native-kernel-speed conclusion follows from this run.

The PyTorch all-FP4 main result (0.75804%) is close to the previously measured
native Sage3 result (0.75023%) on the same E098 capture, providing a useful
but not bit-exact cross-check. The native measurement is in
`results/research/E098/sage3_grid_ab_sm110_gpu0.json`.
