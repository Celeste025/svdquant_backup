# Low-rank branch: `svd_lowrank` instead of full SVD

## Problem

SVDQuant’s low-rank residual path calls `LowRankBranch.reset_parameters(weight)` during
smooth search (temporary) and during weight-stage low-rank calibration.

The upstream implementation used a **full** `torch.linalg.svd(weight.double())`.
On Flux-style fused projections (e.g. qkv `[9216, 3072]`), that dominates runtime
(~minutes per module). With dozens of modules and grid search, PTQ becomes
impractically slow.

## Change

`deepcompressor/nn/patch/lowrank.py` now uses:

```python
torch.svd_lowrank(weight.float(), q=rank+8, niter=2)
```

then slices the leading `rank` factors into `a` / `b` (same layout as before:
`a`: `[rank, ic]`, `b`: `[oc, rank]`).

## Empirical note (Flux qkv microbench)

On a representative Flux fused qkv weight, `svd_lowrank` was ~1000×+ faster than
full SVD. Smooth α/β search returned the same grid choice in that probe; relative
reconstruction NMSE differed slightly (approx. SVD), which is acceptable for PTQ
smoke / iteration speed. Prefer full SVD only if bit-matching a published checkpoint
that was produced with exact SVD.

## Scripts

`scripts/run_ptq_svd_lowrank.py` historically monkeypatched this behavior; with the
in-tree default, it is optional / redundant but still safe to run.
