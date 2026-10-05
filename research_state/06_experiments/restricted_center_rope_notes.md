# Restricted-center RoPE notes

Status: local source inspection and algebra only. No new search, sample fitting, kernel, or GPU execution. E028–E030 do **not** establish that RoPE low-pass filtering explains rank16 query-center structure.

## Verified H3 layout

- Head dimension is 128; `rope_inv_freq_len=16`. Frequencies are `omega_j=10000^(-j/16)`, j=0..15. Forward rebuilds this schedule rather than using the registered `inv_freq` parameter directly. The frequency vector is `[T16,H16,W16,T16,H16,W16]`; split-half rotation therefore pairs T channels `0:16` with `48:64`, H `16:32` with `64:80`, and W `32:48` with `80:96`. Channels `96:128` are 32 unrotated dimensions. Frequencies are shared across heads, and the same position-derived tensor is used across DiT blocks.
- The active Comfy attention path reshapes QKV as `[S,3,H,D]`, applies Q/K RMSNorm, then the same RoPE implementation. Thus the argument concerns post-normalization, post-RoPE Q, not raw projection outputs.

Sources (absolute paths, one-based lines):
- `/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:47` — split-half pairing and pass-through channels; `:84` — frequency construction; `:263` — heads/head dimension; `:275` — frequency count; `:360` and `:383` — shared RoPE tensor across blocks.
- `/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/models/minimax_h3_dit_comfy.py:17` — active QKV layout, normalization, and rotation.

## What averaging can and cannot filter

The actual video grid is `[37,18,32]`, T-major and W-fast, with 576 tokens per latent frame. For latent H=36/W=64, the spatial coordinate spacing is `4/3` along both axes. A complete 128-token group wholly inside one frame traverses four complete cycles of W coordinates, only about 4–5 H rows, and a **constant T coordinate**. Constant-amplitude W components therefore have a geometric cancellation mechanism; H has a shorter averaging span. Even the highest temporal rotary frequency has no within-frame phase variation to cancel. Groups crossing frames instead see the actual nonuniform temporal increments `(1,4,4,4,4) * 5/3`.

The packed order is `[text | cond | audio | video | pad]`; the examined no-reference cases have no cond rows. Text has increasing T and H=W=0. Audio is channel-major: T increases within each channel, H=0, and W is constant within each channel (opposite spatial endpoints distinguish the two channels). Consequently an interior text group or single-channel audio group has 96 dimensions without within-group rotary phase variation: 64 spatial rotary dimensions plus 32 pass-through dimensions. Groups spanning modality/channel/frame boundaries obey neither one common axis nor one uniform 128-point filter. Existing p30/p36 video-start phases also differ; these are coordinate/grouping facts, not a demonstrated source of quality loss.

Sources:
- `/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/models/minimax_h3_dit.py:15` — video flattening; `:32` — channel-major audio packing.
- `/home/wjq/workspace/svdquant-exp/third_party/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:588` — temporal constants; `:611` — spatial grids; `:618` — temporal grid; `:631` — meshgrid/flatten; `:643` — audio channel coordinates; `:647` — packed order; `:666` — text coordinates; `:685` — video/audio coordinates.

## A definable diagnostic selector is not an explanation

Without reading Q values, one can compute, for each rotary pair and actual Q group, `z_g = mean(exp(i * omega * position))`, using the actual valid tokens and the same zero-padding denominator as Qmean. A geometry-only diagnostic could rank complete complex pairs by the across-group variance of `z_g` after removing its global mean, with a fixed rank budget and deterministic tie handling. This uses metadata, not same-sample PCA. It is more relevant to `mu_block - mu_global` than simply choosing the lowest frequencies: a nearly constant phase component may survive each block mean but be absorbed by the global center.

However, this score assumes coherent pre-RoPE amplitudes with comparable importance across channels. For independent zero-mean isotropic Q, orthogonal rotation does not change block-mean variance; position-correlated Q may also counteract rotary phase. Real learned channel amplitudes/correlations and K quantization-error geometry are absent from the score. It therefore predicts only a specified constant-amplitude model, not actual centering or attention benefit.

The unrotated `96:128` subset is a structurally defined rank32 diagnostic; selecting 16 of those dimensions has no source-derived preference. A coordinate basis would make `BKc^T` a channel gather rather than a GEMM, but its accuracy remains unmeasured. The source does not justify a unique, useful rank16 selector or the narrative that low frequencies automatically explain the observed rank16 signal. No method or causal mechanism is adopted from this inspection.
