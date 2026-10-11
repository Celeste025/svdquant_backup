# E087: fast timestep-aware smoothing probe

## Scope

This one-hour exploratory probe used MiniMax-H3 block 13, the `qkv_proj` and
`out_proj` linears, 20 denoising timesteps, and the fixed ten-point smoothing
grid.  Prompts 1 and 3 selected a smoothing vector; disjoint prompts 11 and 12
evaluated it.  The comparator is one smoothing vector selected globally over
all timesteps from the same two fitting prompts.  The objective is parent DiT
block-end W4A4 NMSE, with exactly one candidate linear quantized at a time.

## Result

The selected alpha does depend strongly on timestep in both layers: each layer
uses every alpha in 0.1--0.9 across the 20 steps.  This establishes that the
training-side optimum is not time-invariant on this grid.

That variation is not uniformly a held-out improvement.  For `qkv_proj`, the
per-timestep selection has -0.78% pooled held-out NMSE gain versus global
smoothing (11 steps better, 9 worse); the mean per-step gain is -5.97%.  Its
vectors are nevertheless close to global (mean cosine 0.987; minimum 0.956).

For `out_proj`, per-timestep selection has +10.82% pooled held-out NMSE gain
(14 steps better, 6 worse), though individual steps range from -46.0% to
+49.2%.  The selected vectors differ materially from global smoothing (mean
median absolute log ratio 1.025, mean cosine 0.922; minimum 0.850).  Pooling
both selected linears gives +4.88% NMSE gain, dominated by `out_proj`.

## Decision

Timestep-aware smoothing is a plausible *layer-selective* direction, but this
probe is not evidence for deploying an independently fitted vector at every
timestep: benefits do not transfer consistently to both linears, and the two
held-out prompts make uncertainty large.  The next reliable test is a larger
fit/validation split and a regularized or timestep-grouped smoothing scheme,
especially for out-projection-like layers, rather than unconstrained
per-timestep alpha selection globally.
