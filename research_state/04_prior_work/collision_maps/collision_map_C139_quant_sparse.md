# C139 collision map — 2026-10-31

Status: partially verified; no first-of-kind claim.

## C139A

Claim hypothesis: with fixed W4 weights and equal tile/precision budget,
quantization-conditioned masks can reduce end-of-block error more than masks
chosen for dense BF16 attention, because signed sparse and quantization
residuals interact.

Nearest: DraftAttention2 (importance-ranked joint precision/sparsity), SLA2
(learned output-reconstructing router + QAT), Veda (dense geometry distillation),
QuantSparse (joint compression recovery), ALLOT (final-video allocation).
Overlap: learned masks, joint budgets and output losses are already done.
Residual: fixed packed quantized trunk, teacher before upstream projection
quantization, signed marginal action gains, demonstrable compression-conditional
mask inversion that cheap features can predict. Merely replacing attention-mass
loss by MSE is too thin. Risk: oracle freedom/overfitting, negative correction
does not transfer, previous joint QAT already captures it.

## C139B

Claim hypothesis: low-bit draft-score uncertainty causes discrete selection
mistakes that can be repaired using small higher-precision descriptors.
Nearest: Veda statistics-aware scores, DraftAttention2 pooled scores,
ThriftAttention importance proxies, quantized-key Jensen correction.
Overlap: improving routing proxies and selective precision are well covered.
Residual: a measured decision-instability mechanism tied specifically to low-bit
score error, a calibrated uncertainty rule and an equal-latency repair policy.
Risk: routers already use BF16 summaries; upstream W4 dominates instead. In
that case reject rather than introducing needless higher-precision routing.
