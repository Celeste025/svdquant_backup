# E158 — preparation kernel inspection

GPU2, H3 Q128 / 38,661 tokens, mixed NVFP4+INT8, three profiler calls per toggle.
CUDA kernel durations (ms), not full API latency:

| Kernel | Pool off | Pool on |
|---|---:|---:|
| QK prepare |6.726|7.671|
| V prepare |2.031|2.026|
| INT8-path FP8 V from partials |0.874|0.872|
| Sum |9.631|10.569|

Pooling is fused into the QK input pass; disabling it removes arithmetic and
small pooled outputs, not the principal input scan/quantization work.
Source inspection: QK rereads raw input after block-scale reduction to emit
INT8. packed FP16 Q/K are written even with has_fp16=False. V is written as
FP16, then reread for FP8 conversion. The V conversion's four stage stripes
repeat the global channel-scale reduction. The QK task has 128 threads, one
per channel, traversing 128 tokens serially for Q128. These are optimization
candidates, not measured speedups. Prioritize avoiding unused packed Q/K writes
and a cooperative tiled QK load/quantization design; validate numerical scale,
rounding, masking and output equivalence before accepting changes.
