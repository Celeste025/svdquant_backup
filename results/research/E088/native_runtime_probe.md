# E088 — H3 native low-bit runtime probe

Date: 2026-10-09.  This is a **linear microbenchmark**, not complete DiT
latency.  The real input was exported from the existing teacher-forced H3
cache: prompt 1, denoising step 0, DiT block 25.  It has 22,400 packed tokens.
The first probe below used 1,024 contiguous real tokens because only 6--7 GiB
was available; the later warm-run comparison uses all 22,400 tokens.

## Native-kernel gate

| Candidate | Evidence | Decision |
|---|---|---|
| NVFP4 W4A4 | NVIDIA CUTLASS 79a/79b, compiled `sm_120a`; official reference check passed | eligible |
| INT4 W4A4 cuBLASLt | packed `CUDA_R_4I` x `CUDA_R_4I` smoke test returned `CUBLAS_STATUS_NOT_SUPPORTED` | reject |
| INT4 W4A4 PyTorch | `aten::_dyn_quant_matmul_4bit` has CPU dispatch only | reject |
| INT4 W4A4 CUTLASS legacy IMMA | packed `int4b_t` smoke kernel compiled but launch failed on this Blackwell GPU | reject |

The rejected INT4 W4A4 alternatives must not appear as a GPU deployment
baseline or a speed curve.  TensorRT's deployable INT4 route is weight-only,
not W4A4.

## Replacement measurement protocol

The provisional cross-format table below is **not** the final deployment
comparison.  Its TorchAO FP4 activation-pack number is retained only as a
reference-path diagnostic.  The replacement experiment uses the exact real
H3 `blocks.25.mlp.fc2` tensor pair (`M=22400, K=14336, N=5376`) and runs each
format after warmup with CUDA events:

| Arm | Tested implementation | Counted online work | Excluded offline work |
|---|---|---|---|
| BF16 | cuBLAS BF16 GEMM | GEMM and bias epilogue | none |
| W8A8 | TorchAO dynamic per-row A8 + per-channel W8, `torch.compile` integer path | activation absmax/quant, INT8 GEMM, rescale/bias | W8 pack/scales |
| INT4 W4A16 | TorchAO tile-packed W4A16 CUDA operator | BF16 x packed-INT4 GEMM and rescale/bias | INT4 tile pack/scales |
| NVFP4 W4A4, per-call global scale | FlashInfer CUTLASS SM120 path plus external PyTorch full-tensor amax | full-tensor activation amax, native E2M1 pack, FP4 GEMM/output scale | FP4 pack and scale swizzle for weights |

Each arm will have (a) output NMSE to BF16, (b) steady-state end-to-end
latency, and (c) non-overlapping stage timings.  Warm JIT/compile performance
and first-use compilation latency are separate rows.  A CUDA profiler trace
will bind every stage to actual CUDA kernel names.  The W8A8 arm is accepted
only if its compiled graph still contains an integer GEMM; otherwise it is
reported as unsupported, not silently demoted to BF16.  INT4 is intentionally
labelled **W4A16**, rather than incorrectly called W4A4.  The NVFP4 arm uses
a per-call full-tensor global scale.  TensorRT Model Optimizer and vLLM
normally provide this per-tensor activation scale from offline calibration;
only the block-16 scales remain dynamic in the online quantizer.  Therefore
the 3.098-ms external amax is specific to our selected scale policy and is
not unavoidable NVFP4 deployment overhead.
Likewise, the TorchAO tile-packed W4A16 CUDA operator is a real low-bit
operator, but its result must not be treated as the performance of a mainstream
AWQ/GPTQ/Marlin deployment backend.  The BF16 and W8A8 results are valid
native-operator measurements, not whole-engine or universally optimal numbers.

### First full-shape, warm-run result

Completed 2026-10-09 on an idle GPU 1, with 50 warmups and the median of 11
CUDA-event batches (50 repetitions each; W4A16 uses five due to its much longer
run time).  This is one real H3 FC2: `blocks.25.mlp.fc2`, BF16 weight
`[5376, 14336]`, real cached input `[22400, 14336]` (prompt 1, step 0, block
25).  All output metrics use the same BF16 `x @ W.T` reference.  The W8A8
module was `torch.compile`d before timing; its compilation latency is excluded.

| Arm | Warm median ms | CV | NMSE | Output relative RMS error (`sqrt(NMSE)`) | Status / interpretation |
|---|---:|---:|---:|---:|---|
| BF16 | 14.534 | 1.36% | 0 | 0.00% | PyTorch/cuBLAS BF16 baseline |
| W8A8 | 9.825 | 0.068% | 0.002328 | 4.82% | compiled TorchAO dynamic per-row A8, per-channel W8 integer path |
| INT4 W4A16 | 226.854 | 0.250% | 0.007948 | 8.92% | actual TorchAO tile-packed CUDA operator, but evidently not a performant SM120 kernel at this shape |
| NVFP4 W4A4, per-call global scale | 7.983 | 0.021% | 0.009643 | 9.82% | FlashInfer CUTLASS native FP4 GEMM plus external PyTorch global amax; not a calibrated-scale deployment benchmark |

The W8A8 arm passed a separate CUDA-trace gate: the compiled graph emitted
`cutlass_80_tensorop_i16832gemm_s8_128x128_64x6_tn_align16`, preceded by Triton
per-row activation quantization kernels.  It is therefore not a BF16
dequantize-and-GEMM fallback.

### W8A8 / W4A16 deployment-backend audit (2026-10-14)

The measured W8A8 and W4A16 rows above are **measurements of the named TorchAO
operators**, not measurements of vLLM.  The distinction matters especially
for W4A16.  The deployment comparison target and the existing measurement are:

| Format | Existing E088 path | vLLM-style deployment target | Alignment of existing latency |
|---|---|---|---|
| INT8 W8A8 | dynamic per-row activation INT8, per-output-channel weight INT8, compiled integer GEMM, separate layout/output work | `scaled_int8_quant` dynamic per-token activation INT8; prepacked INT8 weight; `cutlass_scaled_mm` with activation/weight scales and optional bias in the GEMM call | Same broad quantization granularity and native INT8 arithmetic, **not** the exact deployed quantizer/GEMM/epilogue chain. The 9.825-ms result is a TorchAO baseline only. |
| INT4 W4A16 | TorchAO `TILE_PACKED_TO_4D`, `aten::_weight_int4pack_mm`; BF16 input | offline GPTQ-layout **group-128 INT4** packing/repacking; vLLM Marlin weight-only GEMM with BF16 input and group scales | Both use genuine packed INT4 weights, but the kernel, packing format, and quantization recipe differ. The 226.854-ms result **must not** be used as Marlin or general W4A16 deployment latency. |

The audited W8A8 trace contains **1.267 ms** of layout/output-copy kernels in
addition to activation quantization and integer GEMM.  A deployment-equivalent
comparison must time the actual vLLM `scaled_int8_quant` +
`cutlass_scaled_mm` sequence, including its output scaling/bias behavior,
rather than subtracting this copy time by hand.  The existing W8A8 NMSE
(`0.002328`) remains valid for its own TorchAO quantization recipe; it is not
an accuracy result for vLLM's weight packing and rounding.

For W4A16, a replacement must specify the *INT4* recipe (at least group size,
symmetry/zero-point convention, and whether GPTQ/AWQ weight correction is
used), pack/repack the same H3 FC2 weight offline in that format, and measure
the online Marlin BF16-input GEMM.  Weight preparation belongs outside the
warm inference timing.  Its output NMSE must be recomputed for that same
recipe; the TorchAO `0.007948` NMSE cannot silently be carried over.  An
NVFP4 W4A16 GEMM is a different weight format and is not an INT4 substitute.

No vLLM/Marlin package or compiled Marlin extension was found in the existing
project Python environments during this initial audit; an isolated deployment
operator environment and provisional follow-up were added below.  The old
TorchAO rows must still be retained only as named-operator diagnostics.  In
particular, the large TorchAO W4A16 time is evidence about that operator on
this H3 shape, not evidence that deployed weight-only INT4 must be slow.
Conversely, Marlin's use in vLLM does not imply it will beat BF16 at
`M=22400`; that requires a stable isolated measurement on this shape.

Implementation sources checked: vLLM's
[`CutlassInt8ScaledMMLinearKernel`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py),
[`apply_gptq_marlin_linear`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils.py),
and its [INT4 W4A16/G128 deployment recipe](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/llm_compressor/int4.md).

### Deployment-operator replacement: vLLM 0.26.0 (2026-10-14)

`scripts/research/bench_h3_vllm_int8_int4.py` now tests actual vLLM
operators on the same FC2 weight and real 22,400-token input.  The isolated
vLLM 0.26.0 environment is on data1 at
`/data1/models/svdquant-wjq/research/envs/vllm-linear-bench-20261014`;
existing project environments were not modified.  This vLLM release matches
the available PyTorch 2.11 binary.

The intended vLLM CUTLASS INT8 GEMM rejected the RTX PRO 5000 Blackwell
(`SM120`) at runtime with `Int8 not supported on SM120`.  The deployed vLLM
fallback for this hardware is its `TritonInt8ScaledMMLinearKernel`: online
`scaled_int8_quant` (dynamic symmetric per token) followed by
`triton_scaled_mm` (prepacked symmetric per-output-channel INT8 weight,
scales and BF16 output).  This is the path now benchmarked, **not** CUTLASS
INT8.  The INT4 W4A16 path calls vLLM `marlin_gemm` on offline-repacked
`uint4b8` weights with K-group-128 scales; input remains BF16.  Its offline
weight recipe is round-to-nearest, not Hessian-corrected GPTQ, so its NMSE
must not be interpreted as GPTQ's accuracy.  Offline quantization/repacking
is excluded from online latency in both arms.

The 64-token correctness smoke test passed.  Two separate full-shape GPU6
runs produced the following **provisional, contention-affected** data (20
warmups, seven batches of ten CUDA-event repetitions); all NMSE values use
the same BF16 FC2 output reference:

| Actual operator | Online median | CV | Output NMSE | Status |
|---|---:|---:|---:|---|
| BF16 `x @ W.T` (vLLM-paired run) | 29.810 ms | 3.05% | 0 | GPU6 shared; not an accepted isolated baseline |
| vLLM W8A8 activation quantize | 1.896 ms | 0.05% | — | component of the next row |
| vLLM W8A8 Triton GEMM, prequantized input | 8.168 ms | 0.02% | — | component of the next row |
| vLLM W8A8 full online path | 10.228 ms | 4.38% | 0.002319 | complete path; fails 3% stability gate |
| vLLM **INT4 group-128** W4A16 Marlin full path (not NVFP4) | 35.493 ms | 0.22% | 0.010438 | numerically valid; shared-GPU speed only |
| BF16 `x @ W.T` (FlashInfer-paired run) | 30.032 ms | 9.37% | 0 | separate shared-GPU run; fails stability gate |
| FlashInfer **NVFP4 block-16** W4A16 CuTe GEMM (not INT4) | 61.177 ms | 29.90% | 0.006391 | numerically valid; timing fails stability gate |

The component medians and complete-path median are independently timed and
need not sum exactly.  Marlin versus its own dequantized INT4 weight reference
has NMSE `4.91e-6` on the layout-gate tokens; that check guards against
silently mispacked weights.  The TorchAO W4A16 `226.854 ms` and Marlin
`35.493 ms` are **different recipes and operators**, not paired speedups.
GPU6's other resident workload became active during/after this run, so none
of these latencies should be called isolated deployment performance or used
for a definitive BF16 speed ratio.  Raw JSON and run log are at
`results/research/E088/vllm_deploy_int8_int4_p1_s00_gpu6.json` and
`results/logs/e088_vllm_deploy_fc2_20261014.log`.

The additional **NVFP4 W4A16** arm is implemented in
`scripts/research/bench_h3_flashinfer_nvfp4_w4a16.py`, using FlashInfer
0.7.0.post1's native `mm_bf16_fp4` with the CuTe DSL backend on SM120.
Weight is E2M1 FP4 with E4M3 block-16 scales and an offline weight-global
scale, matching the weight quantization recipe of the W4A4 probe.  Input is
BF16: **there is no online activation quantization, pack, or activation max
reduction** in this W4A16 path.  `nvfp4_quantize` and
`prepare_bf16_fp4_weights` are offline weight preparation, excluded from
the online time.  The required CuTe DSL 4.6.2 package was installed in an
isolated data1 dependency directory; FlashInfer and other Python environments
were not modified.  Its 64-token smoke test passed with NMSE `0.006202`.
The full-shape NMSE `0.006391` is about **38.8% lower relative** than this
one input's round-to-nearest INT4 group-128 NMSE `0.010438`.  That is an
accuracy comparison of two distinct weight quantization recipes, **not** a
speed claim or a generalization across H3 layers.

The NVFP4 W4A16 full-shape GPU6 run had BF16 CV `9.37%` and GEMM CV
`29.90%` under third-party GPU activity, so its `61.177 ms` median is not
an accepted isolated performance result.  JSON and logs are
`results/research/E088/nvfp4_w4a16_p1_s00_gpu6.json` and
`results/logs/e088_nvfp4_w4a16_full.log`.  A fair speed ranking of INT4
W4A16, NVFP4 W4A16 and W4A4 still requires repeated measurements during
an idle GPU window.

The full-shape W8A8 trace breaks its 9.622 ms of serial GPU kernel work into
**1.601 ms** dynamic per-row quantization, **6.753 ms** CUTLASS INT8 GEMM,
and **1.267 ms** layout/output-copy kernels.  The separately measured 9.825
ms warm full-path median remains the latency result; trace kernel-work sums
can include profiling effects.  BF16 and W4A16 have no online activation
pack in these definitions; their warm full-path medians are 14.534 and
226.854 ms, respectively.

For the NVFP4 arm the independently timed stages were BF16 input absmax
reduction **3.098 ms**, native E2M1 pack **0.831 ms**, and FP4 GEMM **4.112
ms**.  Their 8.041-ms arithmetic sum and the 7.983-ms full-path median
are close but not expected to be identical: separate timing changes allocator
reuse and inter-kernel scheduling.  The main result is the complete warm
path.  Earlier measurements were taken while the clock/load state was not
controlled and are superseded; the accepted run reduces BF16 directly and
converts only the scalar scale to FP32.  A production-style calibrated-scale
measurement should remove the external full-tensor amax from the timed path,
use a scale derived from independent calibration calls, and re-check output
NMSE for that scale.  The 0.831-ms pack and 4.112-ms GEMM component timings
do not by themselves establish the end-to-end calibrated-scale latency.

## FC2 native NVFP4 result

Layer: `blocks.25.mlp.fc2`, weight shape `[5376, 14336]`, input shape
`[1024, 14336]`.  Weight and activation were packed to NVFP4 block-16 with
TorchAO; GEMM calls `torch._scaled_mm` with `float4_e2m1fn_x2` packed operands
and E4M3 scales.  Eight CUDA-event timed repetitions after three warmups:

| Stage | ms | Share of end-to-end |
|---|---:|---:|
| dynamic activation quantize/pack | 3.3421 | 93.8% |
| native FP4 GEMM | 0.2172 | 6.1% |
| quantize + GEMM | 3.5612 | 100.0% |

The FP4 output NMSE against BF16 `x @ W.T` is `0.0157001`.

The packing stage here is TorchAO's reference CUDA composition rather than a
production fused quantize-pack epilogue.  Consequently the stage split is
useful to show the cost that fusion must remove, but is not a claim about full
deployment throughput.

## Deployment-runtime follow-up (FlashInfer 0.7.0.post1)

The machine already has FlashInfer 0.7.0.post1 in
`/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002`.  Unlike the
TorchAO packing composition above, its source exposes a native SVDQuant
inference chain for Blackwell:

1. `nvfp4_quantize_smooth(x, pre_quant_scale, global_scale)` is one SM120
   CuTe-DSL CUDA kernel that applies channel smoothing and writes packed E2M1
   plus UE4M3 block-16 scales; it does **not** materialize a smoothed BF16
   activation.
2. `mm_nvfp4_svdquant(...)` is the native NVFP4 residual GEMM and, on SM120,
   fuses the BF16 rank-32/64/96/128 correction and optional bias into the
   GEMM accumulator epilogue.
3. `svdquant_linear(...)` composes the preceding quantizer, the BF16
   low-rank-down GEMM, and the fused residual/correction GEMM.  Weight packing
   and low-rank transforms are offline preparation, not steady-state latency.

A real SM120 plain-W4A4 smoke test succeeded through FlashInfer's CUTLASS backend:
`M=128, K=256, N=512`, packed E2M1 operands and block-16 scales, output BF16;
NMSE to BF16 was `0.0181662`.  This establishes that this runtime uses an
actual FP4 Tensor-Core path, not a BF16 QDQ fallback.  However, the installed
environment lacks the optional CuTe DSL package: on SM120,
`nvfp4_quantize_smooth(..., backend="auto")` has no usable backend and
`backend="cute-dsl"` explicitly reports that CuTe DSL is unavailable.  Thus
the present environment can benchmark **plain NVFP4 W4A4**, but cannot yet run
FlashInfer's fused SM120 SVDQuant chain.  Its optional `b12x` FP4 GEMM backend
is unavailable for the same reason.

Therefore the former NVFP4 timing row below is a **reference-path diagnostic
only**.  A deployment-style *plain W4A4* linear benchmark can use the installed
FlashInfer CUDA quantizer and CUTLASS GEMM without CuTe DSL, with a calibrated
activation global scale provided before inference.  An SM120 benchmark of the
fused SVDQuant chain would additionally need CuTe DSL in this environment.
Neither path should count a per-call full-tensor global reduction as obligatory
NVFP4 inference work.  Offline weight packing and cold JIT are separate from
warm steady-state latency.

## Cross-format profile (same FC2 sample)

All rows use the same real H3 FC2 weight and the same first 1,024 real packed
tokens.  CUDA-event timings use ten repetitions after three warmups.  `N/A`
means that the claimed native CUDA kernel did not initialize and is **not** a
zero or an emulated value.

| Format / path | Input quantize | GEMM | Output rescale | End-to-end | Output NMSE | Kernel status |
|---|---:|---:|---:|---:|---:|---|
| BF16 | N/A | 1.466 ms | N/A | 1.466 ms | 0 | PyTorch BF16 GEMM |
| INT8 W8A8 | 0.948 ms | 0.629 ms | included in full path | 1.878 ms | 0.002614 | `aten::_int_mm` CUDA |
| INT4 W4A16 | N/A | 22.368 ms | N/A | 22.368 ms | 0.007808 | `aten::_weight_int4pack_mm` CUDA; TorchAO `TILE_PACKED_TO_4D` |
| NVFP4 W4A4 | 3.342 ms | 0.217 ms | fused in GEMM output | 3.561 ms | 0.015700 | native `torch._scaled_mm` FP4 Tensor Core |
| INT4 W4A4 | N/A | N/A | N/A | N/A | N/A | no validated GPU kernel (see gate above) |

For INT8, quantization and GEMM are separately measured; the 1.878 ms full
path includes both plus output scale application, so it is the appropriate
comparison to NVFP4's end-to-end number.  For NVFP4, the non-fused reference
quantizer accounts for roughly 94% of end-to-end time.

The first W4A16 attempt used TorchAO's default `PLAIN`/MSLK path and failed
initialization.  The corrected row above selects `TILE_PACKED_TO_4D`, which
does dispatch to the ATen CUDA packed-weight operator.  Its 22.368 ms timing
is nevertheless not a competitive Blackwell deployment result: it is this
particular legacy ATen/TinyGEMM implementation on this shape, and is reported
as such rather than being generalized to INT4 weight-only inference.

## vLLM-style calibrated-scale FC2 follow-up (2026-10-14)

`scripts/research/bench_h3_vllm_style_nvfp4.py` uses the same real
`blocks.25.mlp.fc2` BF16 weight and full `[22400,14336]` input as the warm
comparison above.  A separate cached call (prompt 11, step 0) was replayed
once to supply the **offline** activation global scale; evaluation is prompt
1, step 0.  At runtime the FlashInfer CUDA quantizer computes dynamic
block-16 activation scales and packed E2M1 codes, then its CUTLASS FP4 GEMM
consumes the prepacked weight and writes BF16.  No per-call full-tensor amax
is in the timed path.  This mirrors vLLM's `FlashInferCutlassNvFp4LinearKernel`
operator sequence, but is a standalone linear benchmark, not a vLLM engine.
One calibration call establishes the call-chain and accuracy control; it is
not a production-calibrated multi-sample scale recipe.

| GPU | BF16 median | FP4 quantize median | FP4 GEMM median | FP4 full median | FP4 full CV |
|---|---:|---:|---:|---:|---:|
| 5 | 21.237 ms | 1.801 ms | 9.202 ms | 9.117 ms | 33.86% |
| 4 | 18.153 ms | 1.718 ms | 8.936 ms | 5.570 ms | 24.40% |

The output NMSE is **0.0104245** with the independent fixed scale.  With
exactly the same quantizer, packed weight and GEMM but an evaluation-input
derived global scale (accuracy control only), NMSE is **0.0096432**, matching
the earlier dynamic-scale result.  The fixed pilot scale thus raises NMSE by
about 8.1% *relative*, not 8.1 percentage points.

**Do not accept the latency medians as isolated deployment performance.**
GPU 4 and 5 both carried continuously active third-party jobs during this
run.  The 24--34% full-path CV, BF16 medians above the earlier 14.534 ms idle
result, and the stage/full ordering across non-overlapping timing windows all
expose contention.  The fastest observed complete batches were 5.028 and
4.920 ms, respectively, but even those are not certified idle measurements.
An exclusive or demonstrably idle GPU window is needed for a valid final
speed ratio.  Raw paired run JSON and logs are retained.
