# E102: H3 FC activation global-scale paired test

- Experiment ID: E102.
- Mode: exploration.
- Problem addressed: Does a small dynamic FP32 tensor-wise scale materially improve the local output of a real H3 SVDQuant FC linear compared with fixing that scale to one?
- Hypothesis: At an activation scale near 0.001, the dynamic scale may improve local E4M3 block-scale resolution and lower SVDQuant FC output NMSE.
- Decision informed: Whether a static global scale is a promising deployment simplification, or whether further granularity/scale studies are needed.
- Minimum valid setup: Capture the full real `blocks.0.mlp.fc1` input from prompt 1, step 0, and its BF16 weight. Use the same rank-32 smoothing, low-rank tensors, quantized residual weight, and BF16 `Wx` reference for dynamic versus fixed-one activation global scale. Recompute local block scales dynamically in both arms.
- Invalid cheap proxy to avoid: Random inputs, sampled tokens that alter the global absmax, changing weights or low-rank branch between arms, comparing against a different BF16 target.
- Expected outcomes: Dynamic clearly better; similar within a few percent relative NMSE; or fixed-one better.
- Decision for each outcome: If clearly better, retain dynamic global scale for accuracy; if similar, treat fixed-one as a viable accuracy simplification for this layer only; if fixed-one better, audit FP8 scale rounding and collect additional layers before generalizing.
- Stop condition: One complete paired FC result with finite activation and output NMSE, saved full input and weight, and verified scale near 0.001; do not extrapolate one layer to all 200 linears.
- Evidence type: Necessary for the narrow local-layer question; insufficient for whole-model or generation conclusions.
