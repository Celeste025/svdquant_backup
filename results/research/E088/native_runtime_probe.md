# E088 — H3 native low-bit runtime probe

Date: 2026-10-09.  This is a **linear microbenchmark**, not complete DiT
latency.  The real input was exported from the existing teacher-forced H3
cache: prompt 1, denoising step 0, DiT block 25.  It has 22,400 packed tokens;
the benchmark uses the first 1,024 contiguous real tokens because all GPUs had
only 6--7 GiB free beside external services.

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
format after warmup with CUDA events and preallocated outputs:

| Arm | Deployment definition | Counted online work | Excluded offline work |
|---|---|---|---|
| BF16 | cuBLAS BF16 GEMM | GEMM and bias epilogue | none |
| W8A8 | TorchAO dynamic per-row A8 + per-channel W8, `torch.compile` integer path | activation absmax/quant, INT8 GEMM, rescale/bias | W8 pack/scales |
| INT4 W4A16 | TorchAO tile-packed W4A16 CUDA operator | BF16 x packed-INT4 GEMM and rescale/bias | INT4 tile pack/scales |
| NVFP4 W4A4 | FlashInfer CUTLASS SM120 path | activation global absmax, native E2M1 pack, FP4 GEMM/output scale | FP4 pack and scale swizzle for weights |

Each arm will have (a) output NMSE to BF16, (b) steady-state end-to-end
latency, and (c) non-overlapping stage timings.  Warm JIT/compile performance
and first-use compilation latency are separate rows.  A CUDA profiler trace
will bind every stage to actual CUDA kernel names.  The W8A8 arm is accepted
only if its compiled graph still contains an integer GEMM; otherwise it is
reported as unsupported, not silently demoted to BF16.  INT4 is intentionally
labelled **W4A16**, rather than incorrectly called W4A4.

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
| NVFP4 W4A4 | 7.983 | 0.021% | 0.009643 | 9.82% | FlashInfer CUTLASS native E2M1 Tensor-Core path; online dynamic activation processing is included |

The W8A8 arm passed a separate CUDA-trace gate: the compiled graph emitted
`cutlass_80_tensorop_i16832gemm_s8_128x128_64x6_tn_align16`, preceded by Triton
per-row activation quantization kernels.  It is therefore not a BF16
dequantize-and-GEMM fallback.

The full-shape W8A8 trace breaks its 9.622 ms of serial GPU kernel work into
**1.601 ms** dynamic per-row quantization, **6.753 ms** CUTLASS INT8 GEMM,
and **1.267 ms** layout/output-copy kernels.  The separately measured 9.722
ms end-to-end value remains the latency result; trace kernel-work sums omit
runtime launch overhead.  BF16 and W4A16 have no online activation pack in
these definitions, so their respective 28.505 ms and 499.441 ms rows are
their GEMM/epilogue paths.

For the NVFP4 arm the independently timed stages were BF16 input absmax
reduction **3.098 ms**, native E2M1 pack **0.831 ms**, and FP4 GEMM **4.112
ms**.  Their 8.041-ms arithmetic sum and the 7.992-ms full-path measurement
are close but not expected to be identical: separate timing changes allocator
reuse and inter-kernel scheduling.  The main result is the complete warm
path.  Earlier measurements were taken while the clock/load state was not
controlled and are superseded; the accepted run reduces BF16 directly and
converts only the scalar scale to FP32.

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

Therefore the former NVFP4 timing row below is renamed in interpretation:
it is a **reference-path diagnostic only**.  The next deployment benchmark
must first provision CuTe DSL, then use FlashInfer's native `svdquant_linear`,
include dynamic global-scale reduction plus the native smooth-quantize kernel,
exclude offline weight packing, and report cold/JIT separately from warm
steady state.

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
