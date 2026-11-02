# E112 — H3 V-guided mixed FP4/FP8 P/V tiles

Experiment ID: E112.

Problem addressed: Does a V-tile FP4-error threshold select a small fraction of
tile GEMMs, and does upgrading both P and V on those tiles materially reduce
the output error of real H3 attention?

Hypothesis: The default 0.10 relative-L2 threshold routes a nontrivial but
limited fraction of KV tiles to FP8; the resulting attention output NMSE lies
between all-FP4 and all-FP8 on the same Q/K numerics.

Decision this informs: Whether a real mixed-precision kernel is worth building,
and whether the threshold or fixed FP8-tile budget should be revised.

Minimum valid setup: Use the existing real H3 block-25 prompt-1 step-0 QKV
capture at 480p/3s. Keep Q/K NVFP4 and the P/V online-softmax semantics common
across all-FP4, all-FP8, and mixed arms. Count routing for every head and KV
tile, and compare the complete attention output (all 9,482 main-sequence query
rows and all 54 short-sequence query rows) with an unquantized PyTorch attention
reference. A small query sample may be used only as a preflight, never as the
final output NMSE.

Invalid cheap proxy to avoid: Random synthetic QKV, V-only error as a surrogate
for output error, or comparing paths that change Q/K precision.

Expected outcomes: A sparse FP8 mask with a large NMSE improvement supports a
deployable prototype; widespread FP8 routing or negligible output improvement
argues for a different routing signal/precision budget.

Decision for each outcome: Continue to native kernel only if the output-error
improvement is substantial relative to FP8 usage; otherwise revise or stop.

Stop condition: One checked real capture, complete routing statistics, and
finite paired output metrics are enough for this exploration stage. Do not
generalize the single-block, single-prompt result to full H3 generation.

Evidence: Necessary — routing map and paired output NMSE. Supporting —
threshold sweep and modality split. Cuttable — generation-quality metrics.
