# E097 — MiniMax-H3 native quantized FC2 on idle GPU 0

Completed 2026-10-14 on NVIDIA RTX PRO 5000 72GB Blackwell. This is a **single-linear steady-state operator microbenchmark**, not a full DiT or vLLM-engine benchmark. The real `blocks.25.mlp.fc2` input is prompt 1, step 0, shape `[22400,14336]`; the BF16 weight is `[5376,14336]`. All output NMSE values compare to the same BF16 `x @ W.T` result. Runs use 50 warmups and 11 CUDA-event batches of 30 calls. The GPU was idle before each run, and the successful measurements passed the predeclared 3% CV gate.

| Format / native operator | Online median ms | Paired BF16 ms | Speed relative to BF16 | Output NMSE | CV (full path) |
|---|---:|---:|---:|---:|---:|
| BF16 cuBLAS | 14.09–14.15 | — | 1.00× | 0 | 0.04–0.18% |
| W8A8, vLLM dynamic per-token INT8 + Triton INT8 GEMM | 8.998 | 14.092 | 1.57× | 0.002319 | 0.014% |
| INT4 W4A16, vLLM Marlin group-128 | 15.983 | 14.092 | 0.88× | 0.010438 | 0.235% |
| NVFP4 W4A4, FlashInfer block-16 online pack + CUTLASS GEMM | 4.971 | 14.113 | 2.84× | 0.010424 | 0.083% |
| NVFP4 W4A16, FlashInfer CuTe DSL GEMM | 55.149 | 14.145 | 0.26× | 0.006391 | 0.020% |

The INT4 W4A16 and NVFP4 W4A16 rows use **different weight formats**. The two weight-only kernels are genuine native operators, but neither is faster than BF16 for this large-token H3 FC2 shape. In particular, do not present CuTe NVFP4 W4A16 as an *efficient* baseline here. No native INT4 W4A4 kernel passed the E088 support gate, so no such speed row is reported.

## Online work split

| Full path | Activation quantize/pack | Prepacked GEMM | Full-path median |
|---|---:|---:|---:|
| vLLM W8A8 | 0.932 ms (10.4% of full) | 8.089 ms | 8.998 ms |
| FlashInfer NVFP4 W4A4 | 0.813 ms (16.4% of full) | 4.172 ms | 4.971 ms |

Components and the full path were timed independently, so their medians need not add exactly. Weight quantization/packing and JIT compilation are offline and excluded. W4A4's per-tensor activation scale comes from the separate prompt-11 calibration call; online block-16 packing is included, but recalculating a full-tensor maximum per call is not. W8A8 uses vLLM's actual dynamic per-token quantizer and Triton INT8 GEMM because its CUTLASS INT8 path rejects this SM120 GPU. Marlin uses offline-packed symmetric group-128 INT4 weights without GPTQ/AWQ error correction, which matters for NMSE interpretation.

GPU-0 diagnostic for the scope of the NVFP4 activation cost (same full input, 50 warmups, 11×30 CUDA-event calls): with a **precalibrated global scale**, block-16 quantize/pack was 0.835 ms (CV 1.01%); `torch.abs(x).amax()` alone was 1.889 ms; recomputing the global scale and then packing was 2.750 ms (CV 0.12%). The production-path 0.813 ms number above is thus **not** a fully per-call-global-dynamic quantization cost. The `abs().amax()` diagnostic is a straightforward PyTorch reduction, not a claim that an optimized fused reduction must cost 1.889 ms. A CUDA profiler showed one native `quantize_with_block_size_tma` kernel for each FlashInfer pack call. Diagnostic output is in `results/logs/e097_nvfp4_dynamic_global_scale_gpu0.log`.

FlashInfer `backend=auto` was also tested for W4A4 and yielded 4.994 ms, essentially the same as explicit CUTLASS; both produced identical NMSE. The alternate NVFP4 W4A16 cuDNN backend could not run because cuDNN is absent from the isolated environment. Its failure is retained in `results/logs/e097_nvfp4_w4a16_cudnn_gpu0.log`; no fallback latency is substituted.

Raw measurements: `vllm_int8_int4_gpu0.json`, `nvfp4_w4a4_cutlass_gpu0.json`, `nvfp4_w4a4_auto_gpu0.json`, and `nvfp4_w4a16_cute_gpu0.json`. Reproduction entrypoint: `scripts/research/run_e097_h3_native_linear_gpu0.sh`. All runs use one input/weight pair; timings and NMSE are not generalization estimates across prompts, timesteps, layers, or complete inference. Operator-call overhead, memory allocation, and graph capture may differ in a fully fused engine.
