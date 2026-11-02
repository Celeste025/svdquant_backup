# Optimization progress through E182

Goal active, incomplete. User steering: optimize only90%NVFP4/10%INT8 now;
do not expand precision combinations. Dense INT8 first capture remains part
of the requested cache algorithm. All subsequent experiments use this budget.

## Accepted baseline

E176 capture warp-partial reduction with fixed4 warp allocation for Q64/Q128,
and E178 Q128/D128/no-pool preparation specialization for mixed NV+INT8 and
INT8 capture. E178 output+route hashes equal original480p/768p target inputs.
Current binary restored from results/local/E182/source_before (same E178
binary also in E180/source_before). E178 timing:480p capture32.652/reuse28.838;
768p capture120.869/reuse98.511 ms. Small preparation gain needs controlled
repeat before stable speedup claim. E176 long capture972.737 ms vs original
1086.633 ms; reuse750.467 ms.

## E177 double-K buffers

Removed post-QK overwrite barrier, alternated two K tiles, issued nextK before
softmax. Bitwise identical.480p reuse28.810 vs baseline~29.0;768p99.263 vs
baseline~99.12. No robust gain; restored .inl and instantiation. Binary retained
results/local/E177/candidate_double_k.so. Do not treat as active source.

## E179 real hardware profiling (available tool found)

/usr/local/cuda-13.0/bin/ncu is Nsight Compute2025.3.1; /usr/local/cuda-13.0/bin/nsys
also exists (earlier command-v searches missed these). Kernel counters usable.
Report E179/90_10_480p.ncu-rep and details.txt. Profiler locks clocks, so its
32ms duration is not directly comparable to normal~22ms main-kernel timing.
Q128 mixed main:255 regs/thread with slight spills; occupancy16.57%,2CTAs,
~2active warps/scheduler,53.36%cycles no eligiblewarp. DRAM2.39%, L2hit96.62%,
compute51.3%; fixed-latency dependencies important.11.32B executed instructions,
971M fusedFP32 and2.659B non-fusedFP32. Both sharedmemory and registers limit.

## E180 rejected occupancy3

Removed512B unnecessary shared scale budget in mixed INT8 path and forced
launchbounds3. Output bitwise equal but massive spills:680B stack,2508B spill
stores/2064B loads (per ptxas),168regs vs baseline255. Reuse111.73/424.59ms,
~4xslower. Restored header, instantiation, binary to E178. Candidate retained
results/local/E180/rejected_occupancy3.so. Could test sharedmemory-only later,
but register limit alone still prevents3CTAs; avoid forced cap.

## E181 rejected Q64 compute with Q128 prep/routes

Python wrapper duplicated each Q128 physical route/count row, duplicated the
INT8 Q scale (same quantization), and called Q64 mixed main. Cached static
expanded IDs. Bitwise identical.30call ABA:480p baseline28.910/29.005,
candidate31.422;768p100.386/100.883,candidate111.545. Not integrated; production
source unchanged. Full script bench_e181_q64_compute.py.

## E182 identity-rescale experiment: promising arithmetic, no net speedup

If new online maximum equals previous, exp2(previous-new)=1, so skip d/ro
rescaling in prepare_nvfp4_probability. Macro enabled only Q128mixed target.
Output+route hashes bitwise equal for480p and768p. Timingsreuse28.868/99.005ms,
no robust improvement. NCU report E182/90_10_480p.ncu-rep, details.txt:non-fused
FP32 reduced2.659B->1.597B (~40%);total instructions11.32B->10.44B. But local
spill requests1.71M->27.93M, noeligiblewarp53.36%->56.76%. Candidate is saved
results/local/E182/candidate_identity_rescale.so; restored E178 header,
instantiation and binary afterward.

## Next concrete angle

Try warp-uniform any-vote around the identity rescale fast path, so the
arithmetic saving does not require divergent per-row accumulator live ranges.
Analyze ptxas spills and NCU as well as fullAPI; exact hash validation remains
required for these algebra-preserving changes. Need controlled original vs
accepted paired measurements and later-step realQKV captures before goal done.
