# E098 — H3 real-input SageAttention operator benchmark (2026-10-14)

## Sage3 online preprocessing versus E097 FC2 NVFP4 packing

An additional same-GPU CUDA-event dissection used the current SM110 Sage3
benchmark build and the captured real H3 main attention segment. E097's FC2
activation was `[22400,14336]` (321,126,400 BF16 elements); each attention
Q/K/V is `[1,56,9482,128]` (67,966,976 elements), so Q+K+V together have
203,900,928 elements, 63.5% of the FC2 input. Sage3 pads each to
`[1,56,9600,128]` before packing. Its FP4 correction `delta_s` is
`[1,56,75,9600]` FP32, another 161.28 MB of output.

| Official Sage3 online stage | CUDA-event mean |
|---|---:|
| K centering | 0.387 ms |
| Q, K, V padding/layout, combined | 1.249 ms |
| Q block-mean | 0.266 ms |
| Correction matmul/cast | 0.477 ms |
| Q, K, V FP4 quantize/pack, combined | 0.564 ms |
| FP4 attention core, in the staged online call | 4.443 ms |
| Crop/contiguous output | 0.266 ms |

The detailed centering/padding/mean/correction entries sum to 2.377 ms and
reproduce the official `preprocess_qkv` output exactly. All six staged
boundaries sum to 7.650 ms, close to the separately measured SM110 public-API
online latency (~7.63 ms). The non-core stages within this staged call total
3.207 ms; this is not all quantization. E097's 0.813 ms counted only FC2's
online NVFP4 pack using a **precalibrated global activation scale**; the actual
Q/K/V FP4 pack here is likewise sub-millisecond. E097's deliberately naive
per-call global-absmax diagnostic instead took 2.750 ms including pack, but
is not the same scaling algorithm as Sage3. The ~2.95 ms difference between
E098's separately timed online API and separately timed core-only calls is
not required to equal 3.207 ms: stage and core-only calls have different
hot-cache/input-order conditions. The staged breakdown is the appropriate
answer for what the online Sage3 call actually does. Staged output matched
the official API exactly (NMSE 0); 25 warmups, 12 batches × 12 calls, GPU 0.
Raw stage means and per-batch CVs:
[`sage3_sm110_preprocess_stages_gpu0.json`](sage3_sm110_preprocess_stages_gpu0.json).
Reproduction: `scripts/research/measure_e098_sage3_preprocess.py`.

## Matched Sage2++ preprocessing comparison

The same captured H3 Q/K/V, GPU 0, noncontiguous HND views, 25 warmups, and
12 batches × 12 CUDA-event calls were also used to dissect the official
Sage2++ SM120 path. Staged output and all quantized intermediates matched
the official API/helper outputs exactly.

| Preprocessing component | Sage2++ | Sage3 SM110 |
|---|---:|---:|
| K mean/centering | 0.160 ms (mean only; subtract fused into K INT8 pack) | 0.387 ms (mean and subtract) |
| Q/K/V layout/padding | 0.371 ms (V only) | 1.249 ms (Q, K, V) |
| Q/K/V quantization/pack | 0.534 ms (Q,K INT8; V FP8) | 0.564 ms (Q,K,V FP4) |
| Q group mean and correction matmul/cast | — | 0.743 ms |
| Output crop/contiguous | — | 0.266 ms |
| **Total staged non-core** | **1.065 ms** | **3.207 ms** |
| Core within the staged online call | 6.215 ms | 4.443 ms |
| **Staged online total** | **7.280 ms** | **7.650 ms** |

These are matched **within-online-call** stage measurements. Separately
timed core-only calls have different cache/order conditions and should not
be subtracted from these stage timings to estimate each individual component.
Raw Sage2++ timings: [`sage2pp_preprocess_stages_gpu0.json`](sage2pp_preprocess_stages_gpu0.json);
reproduction: `scripts/research/measure_e098_sage2pp_preprocess.py`.

The quantized core requires Q/K/V quantization and its expected packed
layouts, but not necessarily the *separate* copies/launches used in these
implementations; layout conversion and packing could in principle be fused
or consumed differently by a redesigned kernel. K centering is optional
mathematically for row-wise softmax, because subtracting the same key mean
shifts every logit in a query row by a constant, but improves numerical
robustness under quantization and is enabled in both measured recipes.
Sage3's Q block mean and `delta_s` correction are coupled: the correction
restores logits lost when subtracting Q's block mean. Disabling them changes
the approximation and requires a separate error/latency measurement. The
Sage3 padding/crop is required by *this native core interface* at sequence
length 9482, not by the mathematical attention operation itself.

Source audit: the Sage3 code comes from the official `thu-ml/SageAttention`
checkout at commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`
(2026-01-17). The isolated SM110 benchmark copy has exactly one source-code
delta versus that checkout: `blackwell/launch.h` replaces the hard-coded
170-SM persistent-grid count with the current device's `multiProcessorCount`.
The official `sageattn3/api.py` and benchmark copy have identical SHA-256
`9aec83f8d7165e3b563ba4fc91f6ca5e09b07ef3445b6db5d1105912e35dfb90`.
This is an official-derived operator with a local launch-configuration fix,
not a locally rewritten preprocessing algorithm.

The finer Sage3 K-centering split is 0.114 ms for `K.mean` and 0.274 ms for
the separate in-place subtraction. Sage2++ takes 0.160 ms for `K.mean` and
fuses subtraction into its K INT8 quantization kernel. Sage3's Q/K/V
padding-layout calls each cost ~0.416 ms; Sage2++ only does the analogous
layout/padding materialization for V (~0.371 ms), because its Q/K quantizers
consume H3's noncontiguous HND input directly and write packed INT8 outputs
of the original sequence length. This is primarily three full-tensor passes
versus one, not evidence of a ~3x worse copy kernel. Sage3's core returns a
padded-sequence output; the public API slices to 9,482 rows and calls
`.contiguous()` (~0.266 ms), whereas Sage2++ allocates its output at the
original Q shape, so no analogous post-copy occurs. These are current-source
facts, not a proof that equivalent fused/prepacked alternatives are available
or preserve the same output error. Sage3 also mutates K in place; the E098
online benchmark excludes its *benchmark-only* K restoration copy.

**Current RTX PRO 5000 Sage3 benchmark baseline:** use the device-sized-grid
build in [`sage3_baseline_manifest.json`](sage3_baseline_manifest.json).
`bench_h3_sage_two_scope.py`, `bench_h3_sage_real_gpu0.py`, and
`check_e098_sage_kernel_only.py` now select it by default. Historical E098
original-grid results below remain unchanged; the original build can be
selected explicitly with `--sage3-dir` or `--sage3-target`. This sets a
**benchmark baseline**, not a global replacement of the installed Sage3
runtime or a claim that full H3 generation has been measured with the patch.

## Sage3 persistent-grid A/B diagnosis on RTX PRO 5000

The official local Sage3 source launches its FP4 persistent kernel with a
hard-coded **170-block** grid, the RTX 5090 SM count. GPU 0 has **110 SMs**.
Nsight Compute confirmed the actual unmodified launch was `(170,1,1)` with
1.55 waves/SM; the FP4 kernel was **not** a BF16/FP8 fallback. We built an
isolated diagnostic copy at
`/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe`
with only that line changed to query `multiProcessorCount`. The original
Sage3 installation was not modified or replaced.

Using the same captured H3 Q/K/V, CUDA-event protocol, and mean-of-12×12
timing as the two-scope test below, an original → changed → original sequence
gave:

| Sage3 build | Online API incl. preprocessing | Online useful TOPS | Core only | Core useful TOPS | NMSE vs BF16 |
|---|---:|---:|---:|---:|---:|
| Original 170-block, first | 8.711 ms | 295.9 | 5.544 ms | 465.0 | 0.7502% |
| Device-sized 110-block probe | 7.638 ms | 337.5 | 4.692 ms | 549.4 | 0.7502% |
| Original 170-block, second | 8.718 ms | 295.7 | 5.556 ms | 464.0 | 0.7502% |

Against the mean of the two original runs, changing the grid reduced core
latency by **15.45%** and online latency by **12.35%**; useful core throughput
rose by **18.28%**, online throughput by **14.09%**. The official API outputs
had identical NMSE before and after the change, and native core output matched
its own public API exactly (NMSE = 0). The timing CVs were <0.3% for Sage3.
Raw runs: [`original_1`](sage3_grid_ab_original_1_gpu0.json),
[`sm110`](sage3_grid_ab_sm110_gpu0.json),
[`original_2`](sage3_grid_ab_original_2_gpu0.json). Build and run recipes are
`scripts/research/build_e098_sage3_sm110_probe.sh` and
`scripts/research/run_e098_sage3_grid_ab.sh`.

Nsight Compute on the changed build confirmed `(110,1,1)`, one wave/SM,
~51.9% SM compute throughput, and ~20.8% achieved occupancy; the original
build had `(170,1,1)`, 1.55 waves/SM, ~41.3% SM compute throughput, and
~20.8% achieved occupancy. The Sage2 FP32 core on this GPU reached ~75.2%
SM compute throughput. Profiling inflates absolute kernel duration, so the
formal CUDA-event times above are the performance comparison. The changed
Sage3 kernel is still limited to one ~100-KiB-shared-memory/168-register CTA
per SM; eliminating the partial wave does not remove those resource/latency
constraints. The A/B establishes that the hard-coded grid caused a material
loss, **not** that it explains the entire paper-vs-GPU speedup-ratio gap.

On a separate 8K B4/H32/D128 synthetic shape, the changed Sage3 core reached
564.6 TOPS versus Sage2 FP32 418.5 TOPS (1.35×), close to the H3 1.34×
ratio. Thus H3's sequence length/head count is not the main explanation.
This is a shape diagnostic, not H3 data; the raw record is
[`sage3_grid_ab_sm110_synthetic_b4h32n8192_gpu0.json`](sage3_grid_ab_sm110_synthetic_b4h32n8192_gpu0.json).

## Two-scope deployment-vs-paper follow-up (no timed K clone)

The follow-up [`sage_h3_two_scope_no_clone_gpu0.json`](sage_h3_two_scope_no_clone_gpu0.json)
uses the same real H3 9,482-token main segment but passes the captured Q/K/V
as H3-style **noncontiguous HND views**, with no pre-timed `.contiguous()`.
The online arm calls the unmodified official Sage APIs, including their
per-call K/Q centering, INT8/FP8 or FP4 quantization, layout preparation,
and attention. Sage3 modifies K in place, so a disposable K buffer is restored
from the original **before each CUDA start event**. The benchmark-only restore
is outside the timed region; the official preprocessor itself is inside.
Sage2 and 2++ use the original K directly (verified unchanged). The native
core-only arm receives tensors preprocessed once by the official APIs and
times only the attention kernel, matching the paper's kernel-only scope.

Each row is the arithmetic mean of 12 batches × 12 per-call CUDA-event
measurements after 25 warmups; CV is across the 12 batch means. Same useful
operation count for every row: `4 × 56 × 9482² × 128 = 2.577851465728e12`
(MAC = 2 operations). BF16 has no separate QKV quantization, so its two scopes
are identical rather than being remeasured independently.

| Operator | Online API incl. preprocessing | Online useful TOPS | Core only | Core-only useful TOPS | Online/core batch-mean CV |
|---|---:|---:|---:|---:|---:|
| BF16 PyTorch SDPA | 12.339 ms | 208.9 | 12.339 ms | 208.9 | 0.40% / 0.40% |
| Sage2 FP32 path, INT8 QK / FP8 PV | 7.154 ms | 360.4 | 6.273 ms | 410.9 | 0.16% / 0.50% |
| Sage2++ | 7.318 ms | 352.3 | 6.436 ms | 400.5 | 0.09% / 0.07% |
| Sage3 FP4 | 8.716 ms | 295.8 | 5.542 ms | 465.1 | 0.11% / 0.21% |

Every native-core result reproduces its official online-API output exactly
(measured output NMSE = 0). This is still an attention-operator microbenchmark:
it excludes the H3 54-token tail and output assembly, QKV projection/norm/RoPE,
and the rest of the block. Resetting K outside the event estimates an API call
receiving a fresh disposable K from upstream; it is **not** permission to
cache/reuse preprocessed QKV across layers or timesteps. The older E098 table
below included a timed `K.clone()` and pre-timed contiguous HND materialization,
so its online numbers are a different measurement boundary.

The source is an actual BF16 H3 prompt-1, denoising-step-0, block-25
attention call at 480×832 and 73 frames (73/24 ≈ 3.04 s). The captured
post-RMSNorm/RoPE Q/K and V have shape `[9536, 56, 128]`, split by the
model's actual `cu_seqlens=[0,9482,9536]`; 8,580 tokens are video tokens.
The capture is local-only at
`/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt`.

GPU: NVIDIA RTX PRO 5000 72GB Blackwell (GPU 0). Official SageAttention
2.2.0 (2++, SM120 dispatch), its explicit pre-2++ FP32 accumulation path,
and official SageAttention3 Blackwell native extensions were used. The
explicit FP32 path keeps per-warp INT8 QK / FP8 PV fixed and switches
`pv_accum_dtype` from `fp32+fp16` to `fp32+fp32`. It is **not** the
per-thread INT4 variant described in the SageAttention2 paper; that
kernel is not exposed in this installed 2.2.0 code path. CUDA-event
latency is the median of 11 batches of 10 calls after 20 warmups;
profiler was run separately after formal timing.

| Operator | Main 9,482-token segment | Main effective throughput | Full helper incl. 54-token segment and assembly | Main-output NMSE vs BF16 | Full-output NMSE vs BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 PyTorch SDPA (FlashAttention backend) | 12.269 ms | 210.1 TOPS | 12.643 ms | 0 | 0 |
| SageAttention2 FP32 path, INT8 QK / FP8 PV | 7.362 ms | 350.1 TOPS | 8.028 ms | 0.0212% | 0.0211% |
| SageAttention2++ INT8 QK / FP8 PV | 7.490 ms | 344.2 TOPS | 8.156 ms | 0.0210% | 0.0209% |
| SageAttention3 FP4 | 8.897 ms | 289.8 TOPS | 9.192 ms | 0.7502% | 0.7464% |

"Main effective throughput" is the **public-API throughput**, counting
the time for the full `sageattn`/`sageattn3_blackwell` call. It counts only
the useful QKᵀ and PV matrix
work, with one multiply-add counted as two operations:
`4 × 56 heads × 9482² tokens² × 128 head_dim = 2.577851465728 × 10¹² ops`.
Dividing this *same* workload by each main-segment wall time gives the
TOPS above. This is an effective operation-equivalent rate, **not** the
kernel's actual executed instruction count or GPU peak utilization:
softmax, quantization, copies, and the Sage3 correction are not in the
numerator, while their time is in the denominator. Full-helper latency
is shown separately, so the TOPS column must not be interpreted as its
throughput.

## Audit against paper-reported TOPS

The earlier TOPS column was **not comparable to the paper charts**. The
[official SageAttention README](https://github.com/thu-ml/SageAttention/blob/main/README.md#performance)
explicitly says that its plotted TOPS include only the attention kernel,
excluding quantization and smoothing. To check the timing and calls, we
preprocessed the *same real H3 Q/K/V once*, then timed the exact native
cores directly with CUDA events (30 warmups, 11 batches × 20 calls).
Each direct-core output reproduced its public-API output with zero NMSE.

| Native core | Real H3 core-only latency | Real H3 paper-style TOPS | Synthetic B4/H32/N8192/D128 paper-style TOPS |
|---|---:|---:|---:|
| Sage2 FP32 path | 6.251 ms | 412.4 | 419.2 |
| Sage2++ | 6.404 ms | 402.5 | 407.4 |
| Sage3 FP4 | 5.543 ms | 465.1 | 478.0 |

The synthetic shape is the default batch/head/dimension with 8192 tokens
from the official Sage2 benchmark source; it is a **shape diagnostic**,
not an H3 result. Its modest difference shows H3's shape alone does not
explain the paper gap. The independently profiled Sage3 FP4 core was
also 5.544 ms per call, agreeing with the direct CUDA-event result.
For Sage3 on real H3, the public API spends ~3.35 ms beyond the core
(8.897−5.543 ms), versus ~1.11 ms for Sage2 FP32. That is why comparing
our previous full-API 289.8 TOPS with a paper core-only figure is misleading.

The [SageAttention3 paper](https://ar5iv.labs.arxiv.org/html/2505.11594)
reports a peak of 1038 TOPS on an **RTX 5090**, not this RTX PRO 5000.
[NVIDIA lists](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)
2064 AI TOPS for our card versus
[3352 AI TOPS for the RTX 5090](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/).
These are vendor peak AI-TOPS ratings (NVIDIA's PRO 5000 datasheet
explicitly assumes FP4 sparsity), not measured dense-attention throughput,
but establish that the cards are materially different. After correcting
both the timing scope and GPU, the remaining gap to the paper's peak is
**not fully explained** by this audit; an RTX-5090 run with matched
software and shape would be needed to isolate kernel tuning/hardware
effects. The measured core/output agreement and stable CUDA-event timing
provide no evidence that our H3 QKV was passed to the wrong native kernel
or that elapsed time was calculated incorrectly.

The audit source is `scripts/research/check_e098_sage_kernel_only.py`;
the raw results are
[`sage_480p3s_kernel_only_gpu0.json`](sage_480p3s_kernel_only_gpu0.json) and
[`sage_synthetic_b4h32n8192_kernel_only_gpu0.json`](sage_synthetic_b4h32n8192_kernel_only_gpu0.json).

Thus the Sage2 FP32 path is 1.67× faster for the main segment or 1.57×
including the small tail and output assembly. Sage2++ is 1.64× or 1.55×
faster, and Sage3 is 1.38× or 1.38× faster, respectively. The ~1.7%
Sage2-versus-2++ main latency gap is for this one shape and GPU, not a
general ranking. All latency batch CVs are below 0.5% in this run. The
Sage2 and Sage2++ full helpers use H3's `V+1e-8` guard, whereas the main-segment
operator row does not; the BF16 and Sage3 helpers use BF16 SDPA for the
54-token tail. These are **attention-only** comparisons, not an entire DiT
block or full-video-generation throughput. The Sage3 accuracy cost is
materially larger on this single input, so speed alone does not establish
that it is a suitable default for H3.

Approximate per-call CUDA kernel breakdown from an independently profiled
three-call pass (rounding and profiling overhead mean rows need not sum to
the formal latency):

| Component | Sage2 FP32 path | Sage2++ | Sage3 |
|---|---:|---:|---:|
| Main fused attention kernel | 6.08 ms | 6.21 ms | 5.54 ms |
| Q/K or Q/K/V quantization kernels combined | 0.37 ms | 0.37 ms | 0.55 ms |
| Mean/centering or group-mean preparation | ~0.28 ms | ~0.29 ms | ~0.52 ms |
| Input layout/padding kernel (Sage2 family) or device-copy kernels (Sage3) | ~0.37 ms | ~0.37 ms | ~1.13 ms |
| Additional correction BF16 BMM | — | — | 0.25 ms |

The Sage3 profile additionally shows ~0.24 ms device-to-device copy,
~0.33 ms fill kernels, and ~0.22 ms casting copy. The Sage2 and Sage2++
profiles show ~0.24 ms device-to-device copy each. Profiler `aten::` activity entries
duplicate their underlying CUDA kernels, so they must **not** be added
again. The full unsimplified profiles and timings are in
[`sage_480p3s_fourway_main_gpu0.json`](sage_480p3s_fourway_main_gpu0.json) and
[`sage_480p3s_fourway_full_gpu0.json`](sage_480p3s_fourway_full_gpu0.json). Historical
22,384-token E087-call timings in this folder are **not** 3-second
measurements and were excluded from this result.

Reproduction: `scripts/research/capture_e098_h3_480p3s_attention.py` and
`scripts/research/bench_h3_sage_real_gpu0.py`; source build helper
`scripts/research/build_e098_sage2pp.sh`. The E098 decision record is
`research_state/06_experiments/decision_value_plan_E098_h3_sage_attention_gpu0.md`.
