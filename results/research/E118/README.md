# E118 — reduce native mixed Sage3 core spilling

Input is the captured MiniMax-H3 prompt-1, step-0, block-25 attention call
(`h3_p1_s0_b25_480x832_73f_qkv.pt`), shape `[1,56,9482,128]`, noncausal,
on GPU 0 (RTX PRO 5000 72GB). The corrected E116 attention mask and E117
sparse V router are held fixed. At threshold 0.10, 106 of 4,200 V tiles
select E4M3 FP8; all other tiles use NVFP4. The candidate is compiled into
`fp4attn_mixed_inplace_cuda` and does not replace the existing baseline or
the H3 generation runtime.

## Accepted experimental candidate

The selected FP8 PV branch formerly kept a separate 64-float-per-thread FP8
MMA accumulator and an FP4-layout output accumulator live together. The new
`SAGE3_MIXED_INPLACE_FP8_ACC` build aliases the FP8 C-fragment onto the
output fragment and performs its fixed 16-value layout permutation in place
before applying the existing row and V scales. It does not change routing,
precision, softmax, or the pure-FP4 branch.

All timings below are the mean of three independent Python processes. Each
process used 20 warmups and 8 batches of 12 CUDA-event calls. Core timing
excludes Q/K/V preprocessing and V routing. Full online timing includes
preprocessing, dynamic Q/K/V quantization, sparse V routing/packing, core,
and output crop; it excludes only the benchmark's K-buffer restoration,
which is not part of the operator.

| Scope | Corrected pure FP4 | Existing sparse mixed | In-place mixed | Mixed speedup |
| --- | ---: | ---: | ---: | ---: |
| Core, all route bits zero | 4.679 ms | 13.097 ms | 11.434 ms | 12.70% |
| Core, 106 real FP8 routes | 4.679 ms | 15.493 ms | 13.669 ms | 11.77% |
| Full online API, real routes | 7.817 ms | 18.557 ms | 16.966 ms | 8.57% |

The 128-token control and 54-token tail are bitwise identical to the
existing mixed operator in both forced-FP4 and FP8 cases. On the full main
sequence, forced-FP4 output is bitwise identical to pure FP4. With real
routing, the candidate differs slightly from the old mixed output:
NMSE `1.38e-12`, maximum absolute difference `0.03125`, with more than
99.9998% of elements bitwise equal. Candidate-vs-pure FP4 NMSE is
`2.62e-5`, effectively unchanged from the existing mixed path. These
differences are consistent with floating-point evaluation order, not a
changed quantization policy, but the result is not claimed to be bitwise
equivalent on the full sequence.

Nsight Compute full reports (`39` replay passes each) on the all-FP4 route
show local-memory spilling requests falling from `218,170,420` to
`173,751,220` (−20.36%). Both builds report `168` registers/thread.
Nsight's instrumented duration is not used as latency evidence; the
CUDA-event timings above are the speed measurements. The binary `.ncu-rep`
reports remain local under this directory.

## Rejected candidate

Placing the FP8 route in a non-inlined device helper made the full core much
slower: forced-FP4 `13.17 → 109.56 ms` and real routing `15.39 → 112.69 ms`
in its one-process A/B. Calling through a closure caused a large device-call
state penalty, so this variant is not accepted. Its measured JSON is
`noinline_gpu0.json`; it must not be used for inference.

## Reproduction and limits

Build the accepted candidate with
`bash scripts/research/build_e118_sage3_inplace.sh`. Run
`scripts/research/e118_mixed_core_benchmark.py --candidate inplace` and
`scripts/research/e118_inplace_full_api.py`. Raw process-level JSON is
`inplace_gpu0*.json` and `full_api_gpu0_run*.json`.
The candidate shared source header SHA-256 is
`6cdb7586e2a6ecc118a32c4d06382f098b2730afebf5ee88c68a487a10f21681`;
the compiled accepted `.so` SHA-256 is
`1809308dbb1938f97898c73b39805c3bdc4b3ed14ad959220024370e2d4bbc0e`.

This is a single H3 attention shape/capture and one GPU. The online mixed
operator remains about `2.17×` slower than corrected pure FP4, so the
in-place change is a useful but incomplete optimization. It has not been
integrated into H3 video generation or tested across other head dimensions,
other prompts, or a multi-call model runtime.
