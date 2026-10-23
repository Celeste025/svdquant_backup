# E116 — Sage3 noncausal tail-mask repair on real H3 attention

The input is E098's real MiniMax-H3 block-25, prompt-1, step-0 Q/K/V at
480×832, 73 frames. `cu_seqlens=[0,9482,9536]`: the 9,482-token first
segment is the meaningful text/audio/video segment; the 54-token second
segment is H3's independent alignment tail. A 32-token slice is the
aligned-boundary control. All comparisons are noncausal and use D=128.

The pure fixed fork is `third_party/sage3_h3_maskfix`. It preserves the E098
SM110 launch grid and changes only the first noncausal score mask: each
physical K column is mapped through the K pack's within-32 permutation
before comparison with the logical unpadded length. The fixed mixed core is
built as a distinct module, `fp4attn_mixed_maskfix_cuda`, from the patched
source in `third_party/sage3_native_mixed`; its old binary
`fp4attn_mixed_cuda` was not overwritten. The official and E098 source and
binaries were not modified. Causal masking was not changed or validated.

## Correctness

The numerical reference is the E113 native-NVFP4 Triton path with its
default logical-key mask. The `match_sage3_padding_mask=True` branch remains
only as a historical diagnostic that deliberately reproduces the old bug.
Each A/B run executed all three shapes; the fixed outputs were finite.

| Case | Old pure vs logical Triton NMSE | Fixed pure vs logical Triton NMSE | Fixed native mixed vs logical mixed Triton NMSE |
| --- | ---: | ---: | ---: |
| 32 tokens × 4 heads | 0 | 0 | 0 |
| 54 tokens × 4 heads | 0.00646812 | 0 | 0 |
| 9,482 tokens × 56 heads | 1.73854e-5 | 1.13119e-8 | 1.13757e-8 |

The old mask excludes four valid K positions and admits four padding
positions for both nonaligned shapes. The 54-token fixed pure output is
byte-identical to the correct-mask Triton output in
`pure_fixed_short_gpu3.json`.

## Error against BF16

`fixed_three_vs_bf16_gpu3.json` compares the repaired pure Sage3 FP4,
repaired native mixed kernel, and correct-mask Triton mixed kernel on the
*same* captured Q/K/V. The teacher is BF16 noncausal SDPA at scale
`128**-0.5`, and each NMSE uses its output energy as the common denominator.
The main segment is the relevant result for the H3 video sequence.

| Case | Fixed Sage3 FP4 | Fixed native mixed | Triton mixed |
| --- | ---: | ---: | ---: |
| 9,482-token main NMSE | 0.748434% | 0.745237% | 0.745245% |
| 54-token alignment-tail NMSE | 0.071046% | 0.071046% | 0.071046% |

On the main segment the native mixed and Triton mixed variants reduce
NMSE relative to fixed pure Sage3 by 0.4272% and 0.4262%, respectively
(relative, not percentage points). The absolute NMSE reduction is about
0.0032 percentage points. Triton versus native mixed output NMSE is
1.13757e-8, confirming near-identical outputs. The route selected 106 of
4,200 V tiles (2.52%) for FP8. No tile is selected in the alignment tail,
so all three paths agree there. These numbers are one captured block/case,
not an across-layer or across-prompt quality estimate.

The other reference paths were audited separately: the older MixAttention
Triton wrapper masks logical padding columns in `delta_s`; its PyTorch
`_preprocess` was executed on a 54-token input and yielded finite columns
0–53 and `-inf` columns 54–127. The E112 PyTorch mixed reference explicitly
masks score columns `valid:` and completed the real 54-token case with finite
metrics. Neither reference applies the CUDA K physical-row permutation to
its score columns. The E113 native-NVFP4 Triton path was run numerically on
all three shapes above.

## Timing

GPU 3 was idle before timing. Same captured Q/K/V and identical compiler
flags were used for the old and fixed SM110 builds. Each process performed
20 warmups then 8 batches × 12 CUDA-event calls per scope, and the old/fixed
processes were alternated for three independent runs. All per-run batch-mean
CVs were below 1%. `pure_online` includes Sage3 preprocessing and output
crop but excludes the benchmark-only restoration of its mutated K input.
`mixed_core` starts from packed Q/K/V and an already selected V route; it is
an A/B of the kernel only, not mixed deployment latency.

| Scope | Old mean | Fixed mean | Fixed latency change |
| --- | ---: | ---: | ---: |
| Pure Sage3 FP4 core | 4.695 ms | 4.825 ms | +2.77% |
| Pure Sage3 online API | 7.596 ms | 7.753 ms | +2.07% |
| Native mixed core | 15.256 ms | 15.366 ms | +0.72% |

Thus the pure fix is small but *measurably not zero-cost* on this H3 shape;
the mixed-core delta is within the spread of the three A/B pairs. These are
operator results, not end-to-end H3 generation speed. The unrelated
`third_party/SageAttention3_prescale` fork still contains the historical
mask and was not used in this experiment.

`summary.json` checks the sample counts, source pairing, CVs, and numeric
thresholds. Rebuild with `bash scripts/research/build_e116_sage3_maskfix.sh`;
rerun each arm with `scripts/research/e116_sage3_maskfix_ab.py` and aggregate
with `scripts/research/summarize_e116_sage3_maskfix.py`.
