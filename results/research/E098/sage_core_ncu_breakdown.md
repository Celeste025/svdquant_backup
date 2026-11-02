# E098: native Sage attention core breakdown on H3

Input: captured H3 prompt 1, step 0, block 25 main-attention Q/K/V,
`[B,H,N,D]=[1,56,9482,128]`, BF16, non-causal. Device: GPU 0,
RTX PRO 5000 72GB Blackwell (110 SM). Sage3 is the official FP4 core with
only its persistent launch grid changed from fixed 170 CTAs to the runtime
110-SM count. Sage2++ uses its native INT8 QK / FP8 PV core.

| Metric | Sage2++ core | Sage3 FP4 core |
|---|---:|---:|
| CUDA-event mean, 144 calls | 6.430 ms | 4.670 ms |
| Useful throughput, `4BHNN D / time` | 400.9 TOPS | 552.0 TOPS |
| Speedup vs Sage2++ | 1.00x | 1.377x |
| Native kernel launch | 4200 CTAs, 128 threads | 110 CTAs, 384 threads |
| Waves per SM | 19.09 | 1.00 |
| Registers per thread | 255 | 168 |
| Dynamic shared memory per CTA | 32.77 KB | 100.35 KB |
| Theoretical / achieved occupancy | 16.67% / 16.49% | 25.00% / 20.82% |
| Tensor-pipe active / elapsed cycles | 73.62% (FP + INT) | 50.34% (FP4) |
| SM issue-active / elapsed cycles | 35.70% | 43.28% |
| XU / LSU pipe activity | 19.56% / 19.83% | 32.25% / 27.33% |
| DRAM peak throughput | 2.80% | 4.84% |
| L2 hit rate | 96.73% | 93.30% |
| Local register spilling | none | none |

Nsight Compute confirms that Sage3 really executes native FP4 Tensor Core
instructions. Its Tensor pipe is active for only about half of elapsed cycles,
versus almost three quarters for Sage2++; this is the strongest observed
explanation for why the nominal FP4 arithmetic advantage does not appear as a
near-2x **end-to-end core** speedup on this device/shape. This is not an HBM
bandwidth or register-spilling bottleneck. The Sage3 persistent kernel runs
one CTA per SM; each CTA uses about 100 KB shared memory. Nsight reports that
both register and shared-memory requirements cap it at one CTA per SM. Sage2++
can run two CTAs per SM, but its *overall occupancy and scheduler eligibility
are actually lower*, so launch geometry alone is not a demonstrated cause.

The Sage3 fused main loop contains FP4 QK GEMM, delta-score correction,
masking/online softmax, quantization of softmax probabilities to FP4 with
microscales, FP4 PV GEMM, and output rescaling. These stages are interleaved
inside one kernel; pipe utilization percentages overlap and **cannot be added
or translated into milliseconds for individual stages**. Source-counter
sampling of Sage3 found 121,106 fixed-latency-wait samples, 60,820 short-
scoreboard samples, and 35,171 sleeping samples among not-issued warp samples.
This supports substantial instruction/dependency latency, but does not prove
which high-level stage consumes a specific fraction of runtime.

The official paper's RTX 5090 Figure 4 compares Sage3 with *SageAttention2*,
not 2++. On this RTX PRO 5000, Sage2's FP32-accumulation core takes 6.269 ms,
only 2.6% less than the Sage2++ core. Thus baseline naming contributes little
to this local ~1.38x versus paper ~1.8-1.9x gap. Different GPU, sequence
length, and shape still prevent a controlled attribution of the remaining gap.

Timing source: `sage_h3_sm110_two_scope_gpu0.json` (CUDA events, 12 batches of
12 calls, 25 warmups). These timings, not profiler wall times, are used for
speed. Native-kernel NCU reports are stored outside Git at
`/data1/models/svdquant-wjq/research/20261014/E098/` as
`h3_sage2pp_core_breakdown.ncu-rep`,
`h3_sage3_sm110_core_breakdown.ncu-rep`, and
`h3_sage3_sm110_core_source.ncu-rep`. Entry point:
`scripts/research/profile_e098_sage_core_ncu.py`. Both direct core outputs
were checked to have zero NMSE against their corresponding public APIs.
