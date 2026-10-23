# E114 — Native Sage3 mixed-PV baseline and profile (2026-10-21)

**Numerical comparison update:** The E114 native-vs-E113-Triton output NMSE
below was measured before the E115 E2M1 midpoint, K/V traversal-order and
padding-mask alignment fixes. See `results/research/E115/README.md` for the
corrected paired comparison. E114 timing and native-versus-official measurements
remain historical measurements of the implementations at that point.

The implementation target is the original CUTLASS/CUDA SageAttention3
Blackwell kernel, **not** another Triton attention core. The isolated source
copy is `third_party/sage3_native_mixed/sageattn3`; the installed Sage3 used by
E113 remains unchanged. The native mixed branch now passes numerical checks
on real H3 QKV. It is an experimental baseline, **not** yet a deployment-speed
optimization.

The copied CUTLASS/CUDA core builds independently as `fp4attn_mixed_cuda`
with `scripts/research/build_e114_sage3_native_core.sh`. Its all-FP4 output
is bit-identical to installed Sage3 (NMSE 0), including at full 56-head H3
shape. The FP8 path uses a native SM120 E4M3×E4M3 MMA with dynamic group-128
scales for P and V. CuTe's F8F6F4 atom expects **raw FP8 encoding bytes** in
its `uint8_t` fragments; numerically converting E4M3 to uint8 is wrong. The
QK score accumulator also needs a fixed pair-interleave correction before
staging row-major P8. The current 16 KiB P8 tile uses CTA-private global
scratch because the GPU's opt-in shared-memory ceiling is 101,376 bytes and
original Sage3 already uses nearly all of it. The compiled extension's SASS
contains both `OMMA.SF...E2M1` (NVFP4) and `QMMA...E4M3` instructions.

## First native component: V-tile route and FP8-128 preparation

`third_party/sage3_native_mixed/v_tile_router.cu` computes the E113
threshold-0.10 V-FP4 relative-L2 decision once per `(batch, head, 128-key
tile)`, and packs V into E4M3 with one dynamic FP32 scale per output channel
and 128 keys. This is designed for the copied Sage3 PV mainloop to consume.
Build with `scripts/research/build_e114_sage3_native_mixed_router.sh`.

On real H3 QKV at shape `[1,56,9482,128]` (K padded to 9600),
`scripts/research/validate_e114_sage3_router.py` finds 106/4200 FP8 V tiles.
The route mask and all FP8 scales match the PyTorch FP8-group-128 reference
exactly. Three of 68,812,800 FP8 encoded values differ at rounding ties;
the code NMSE is 6.27e-18 and maximum absolute code difference 0.001953125.
The relative-error-square maximum absolute difference is 4.66e-9.
CUDA-event timing with 10 warmups and 30 measurements: median 2.257 ms for
the standalone route+FP8 pack on the RTX PRO 5000 72GB.

The Nsight Compute capture is `router_h56_k9482_full.ncu-rep`, collected via
`scripts/research/run_e114_sage3_router_ncu.sh`. Its replay duration is
2.04 ms, L2 throughput 93.76%, DRAM throughput 7.61%, compute throughput
8.43%, achieved occupancy 63.52%, and 64 registers/thread. Nsight attributes
67.13 MB of excess L2 sectors (86% of global sectors) to uncoalesced accesses.
The likely source is writing transposed `[B,H,D,K]` FP8 with one CUDA thread
per D channel; that causal attribution is a code-based inference to verify by
profiling a different output layout. Long-scoreboard and LG-throttle stalls
also indicate a memory-access bottleneck. This preprocessing alone is about
28% of the prior 8.02 ms Sage3 full-API single-call timing at the same shape,
so it must be reduced or fused before deployment claims.

## End-to-end numerical results

`scripts/research/validate_e114_sage3_mixed_core.py` passes at 128, 256,
1024 and 9482 tokens, including all 56 heads at full length. At 128 tokens,
forcing all PV tiles to FP8 gives output NMSE 0.00324 versus BF16, while
official FP4 gives 0.00855. At 1024 tokens these are 0.00490 and 0.00980.
Forced-all-FP8 is a branch check, not the routing policy. An isolated
128×128 FP8 GEMM gives exactly 128 for all-ones inputs and matches PyTorch
FP32-accumulation matmul with NMSE 0 on random E4M3 operands.

On full real H3 `[1,56,9482,128]`, 106/4200 V tiles select FP8 (2.52%).
The paired report `native_full_attention_profile.json` from
`scripts/research/bench_e114_sage3_native_mixed.py` records:

| Pair, same QKV | NMSE |
| --- | ---: |
| copied all-FP4 vs installed Sage3 core | 0 |
| native mixed vs installed Sage3 FP4 | 0.00002617 |
| native mixed vs E113 Triton mixed | 0.00147649 |
| E113 Triton mixed vs installed Sage3 FP4 | 0.00151270 |

The native/Triton mixed difference is on the same order as the pre-existing
Triton-versus-Sage3 core difference. This is a same-input attention-output
comparison, not a generated-video quality result.

## Speed and bottleneck, 56 heads × 9482 tokens

CUDA-event medians after 10 warmups and 30 repeats on RTX PRO 5000 72GB:

| Component | Median |
| --- | ---: |
| native V route + FP8 pack | 2.289 ms |
| installed Sage3 all-FP4 core | 4.454 ms |
| copied Sage3 all-FP4 core with mixed code compiled in | 12.111 ms |
| native mixed core | 14.891 ms |
| installed Sage3 full API | 7.803 ms |
| native mixed full API | 20.621 ms |
| E113 Triton mixed full API | 272.648 ms |

The native baseline is 13.2× faster than the correctness-first Triton
implementation but 2.64× slower than installed Sage3. Even the copied
all-FP4 core is 2.72× slower despite selecting no FP8 tile. Full-API arms
all use the same BF16 QKV and include preprocessing; core times exclude it.
The router is measured separately, not fused into the core.

Nsight Compute reports were collected with
`scripts/research/run_e114_sage3_native_core_ncu.sh` and saved locally as
`official_fp4_h56_k9482_full.ncu-rep`, `copied_fp4_h56_k9482_full.ncu-rep`
and `native_mixed_h56_k9482_full.ncu-rep` (plus the earlier router report).
NCU replay times are perturbed and not interchangeable with CUDA-event times.

| NCU metric | Official FP4 | Copied FP4 | Native mixed |
| --- | ---: | ---: | ---: |
| replay duration | 6.60 ms | 14.63 ms | 17.23 ms |
| local-memory spilling requests | 0 | 203,503,060 | 248,936,260 |
| local spilling request overhead | 0% | 89.96% | 91.34% |
| registers / thread | 168 | 168 | 168 |
| L2 throughput, peak-normalized | 19.63% | 61.92% | 65.12% |
| DRAM throughput, peak-normalized | 4.89% | 39.67% | 49.19% |
| eligible warps / scheduler | 0.64 | 0.32 | 0.31 |
| achieved occupancy | 20.80% | 20.80% | 20.84% |

The same 168 registers/thread and occupancy do **not** mean register pressure
is unchanged: the FP8 branch adds live state which spills heavily, even when
the runtime mask selects only FP4. Spill traffic explains the large all-FP4
regression. The mixed route adds further spill and scratch traffic. The
standalone V router costs 2.289 ms; its NCU report shows 93.76% L2
throughput and 67.13 MB excess L2 sectors from uncoalesced accesses. A
transposed FP8 output store is the leading code-based suspect, not proven by
an alternative-layout profile.

## Next optimization order

1. Compile an all-FP4 specialization so the FP8 branch is absent from its
   register allocation; check that bit identity remains and core latency
   returns toward 4.45 ms.
2. Shorten FP8-path register live ranges or isolate rare-path code, checking
   SASS local loads/stores and NCU spill requests after each change. An
   out-of-line device helper is an experiment, not assumed to help.
3. Reduce the P8 global-scratch cost only after the spill regression is
   addressed. Reusing output shared memory requires a lifetime/barrier proof.
4. Rework V FP8 pack stores for coalescing; remeasure the full API.

The binary NCU reports, logs and captured tensors remain local/unversioned;
reviewable numerical JSON and this report are under `results/research/E114/`.
