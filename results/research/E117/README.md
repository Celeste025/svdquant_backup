# E117 — native Sage3 mixed-attention cost split and sparse V packing

Input: E098's real MiniMax-H3 prompt-1, step-0, block-25 Q/K/V, main segment
`[1,56,9482,128]`, noncausal, RTX PRO 5000 72GB (GPU 0). The pure and mixed
cores are the E116 corrected-mask builds. All timings use CUDA events and
exclude only the benchmark's restoration of the K input that Sage3 mutates.

## Where the time goes

`e117_native_mixed_cost_split.py` used 20 warmups and 8 batches × 12 calls.
On the real route, 106/4,200 V tiles select FP8, but all 56 heads contain at
least one selected tile. Setting every route bit to zero makes mixed output
bit-identical to pure Sage3 but does not remove the compiled-in FP8 branch.

| Scope | GPU-0 time |
| --- | ---: |
| Corrected pure Sage3 FP4 core | 4.68 ms |
| Mixed core, all routes forced FP4 | 12.81 ms |
| Mixed core, actual 106 FP8 routes | 15.19 ms |
| Original dense V router + FP8 pack | 2.27 ms |

The roughly 8-ms zero-route penalty confirms that FP8 branch presence, not
the few selected FP8 MMAs, is the primary core regression. E114's Nsight
report found heavy register spilling in the mixed-compiled all-FP4 core;
E117's timing split is consistent with that explanation, but it is not a new
Nsight attribution.

## Accepted optimization: pack only selected V tiles

`v_tile_router_sparse.cu` computes the same error and route decision as the
original `v_tile_router.cu`, then only writes E4M3 V and per-channel scales
for selected tiles. Unselected FP8 V/scale entries are intentionally
undefined; they must be used **only** with a consumer that checks the route
before reading. The original dense router and binaries remain available.

`e117_validate_sparse_router.py` checked the 9,482-token main segment and
54-token alignment tail. Route masks, relative-error scores, every selected
FP8 encoding and scale, and all valid attention output elements are bitwise
equal to the original dense router. Three independent runs gave:

| Router | Mean time across runs | Range |
| --- | ---: | ---: |
| Original dense pack | 2.290 ms | 2.290–2.290 ms |
| Selected-only sparse pack | 0.189 ms | 0.189–0.190 ms |

`e117_full_api_benchmark.py` includes Q/K/V preprocessing, dynamic
quantization, route+pack, core, and output crop. It ran 20 warmups and 8×12
event calls per arm, repeated in three processes:

| Full online API | Three-run mean | Range |
| --- | ---: | ---: |
| Corrected pure Sage3 FP4 | 7.828 ms | 7.823–7.830 ms |
| Original dense mixed | 20.452 ms | 20.392–20.497 ms |
| Sparse-pack mixed | 18.336 ms | 18.304–18.378 ms |

Thus sparse packing reduces mixed online latency by 2.117 ms, or 10.35%,
without changing output. The mixed online path is still 2.34× slower than
pure Sage3. This optimization has **not** been wired into the H3 generation
runtime; it is an isolated, validated experimental operator variant.

## Rejected core candidates

* Halving the query tile from 128 to 64 produced wrong output even on a
  128-token control (NMSE 0.305 versus Q128) and an illegal memory access on
  the full input. This cannot be treated as a launch-only change; its source
  edits and binary were removed. The GPU passed a CUDA smoke test afterward.
  See `q64_failed.json`.
* Skipping FP4 P scaling on selected FP8 tiles changed numerical output and
  made the actual-route core slower (15.77 → 16.42 ms in its A/B). Its source
  edits and binary were removed. See `noquant_gpu0.json`.

The next core work should focus on spill removal or on computing rare FP8
contributions outside the FP4 main kernel; neither is implemented here.
Build the accepted router with
`bash scripts/research/build_e117_sage3_router_sparse.sh`. The standalone
benchmark scripts and the three process-level full-API JSON files provide
the reproduction path.
