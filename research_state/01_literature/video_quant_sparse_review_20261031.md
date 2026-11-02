# Video quantization and sparsity review — 2026-10-31

Scope: primary-source, bounded review for H3-first research discovery. No new
GPU experiments are authorized by this discussion. Novelty is partially
verified, not established. E138 rejects plain activation stochastic rounding
on two rollouts, not all error-structure methods.

## Families and coverage

- SVDQuant/Nunchaku: smoothing, low-rank residual weights, fused execution.
- SageAttention3/VC-Attention: low-bit attention arithmetic and operand preparation.
- DraftAttention2: joint skipped/NVFP4/INT8/FP16 head-level allocation, pooled
  QK mass proxy, precision phases sharing online softmax.
  https://arxiv.org/html/2609.32628v1
- SLA2: learned sparse/linear routing, mixture alpha, end-to-end QAT.
  https://arxiv.org/html/2602.12675v1
- QuantSparse: salient multiscale distillation and second-order temporal residual.
  https://arxiv.org/html/2509.23681
- FPSAttention: training-aware FP8/tile sparsity and denoising-step adaptation.
  https://fps.ziplab.co/
- FastH3/UniServe: deployment-oriented QAD and serving (official release evidence,
  not an independently reproduced quality claim).
  https://haoailab.com/blogs/uniserve-fasth3/
- PulseQuant: propagated sensitivity and adjacent weight-code corrections.
  https://arxiv.org/html/2609.33384
- DSAQuant: stage-specific QAT supervision and final-step CFG changes.
  https://arxiv.org/abs/2609.04031
- Veda: distilled full-attention tile geometry, learned sparse masks.
  https://arxiv.org/html/2605.30325
- ALLOT (2026-10-21): final-output-aware layer-step sparsity under measured
  latency. Basic layer-step budget optimization is already directly occupied.
  https://arxiv.org/html/2610.29441
- Quantized Keys Steal Attention: Jensen bias correction in autoregressive
  quantized KV versus current unquantized tokens. This setting differs from
  bidirectional H3, but forbids claiming new softmax-bias correction generally.
  https://arxiv.org/html/2605.26266

## Signals / problem

S1: Existing local error and oracle permutation evidence does not establish a
deployable improvement; E138 lowers adjacent-row error correlation while
raising MSE and both measured video distances. Stop topology as a method claim.
S2: Routing changes a discrete computation graph after upstream linear
quantization. A good dense-reference mask need not be the best mask for a
frozen quantized trunk. This is a hypothesis, not an observed routing defect.
S3: Mass-based objectives ignore signed interaction between fixed quantization
error and a mask change. Generic output-aware selection is already covered;
only quantization-conditioned residual decision value remains a candidate.

## Evaluation

Use matched input/checkpoint/initial latent, fixed retained physical tile and
precision counts first, then measured wall-clock including routing and pack.
Hold prompt/seed and long-sequence evaluation outside calibration. Local
attention/block outputs are feasibility probes, not generation-quality proxies.
Require closed-loop videos, LPIPS/reference distances plus blind quality
review. Compare BF16 and quantized trunks, dense low-bit attention, mass routing,
learned output-aware routing, and random masks with equal budgets.

## Provisional candidates

C139A: Quantization-conditioned signed residual routing. Frozen W4 trunk;
change mask/precision assignments to reduce the end-of-block or DiT residual
to the BF16-trunk teacher, rather than only recovering dense attention on the
already-quantized QKV. Inference needs a cheap predictor; teacher is offline.

C139B: Precision-stable support selection. Quantization of pooled QK/draft
scores may flip borderline Top-k decisions; preserve/recompute only uncertain
routing scores using high-precision small summaries, while keeping expensive
token GEMMs low-bit. Separate low-bit router errors from upstream projection
errors: BF16 summaries of quantized inputs cannot undo upstream W4 errors.

Minimum next test: capture 4 prompts x 3 steps x 3 layers with full QKV;
quantify exact output changes of controlled mask swaps at equal budget.
Split development/test prompts before searching swaps. Route-repair oracle
must reduce pooled full-block SSE >=15% on held-out prompts with improvements
on >=75% calls, then survive full-DiT/closed-loop validation. These are chosen
decision thresholds, not theoretical guarantees. If gains are small or arise
only in stochastic mask searches, stop before training kernels/controllers.

Priority: C139A has greater conceptual upside, C139B is cheaper but potentially
incremental. No method is validated or judged conference-ready.
