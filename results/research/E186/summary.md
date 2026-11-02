# Optimization checkpoint through E186

User requested: finish current experiment and summarize before new experiments.
E186 has finished. E187 trajectory capture scripts uploaded but NEVER launched.
Goal remains active; do not claim deep reuse optimization complete.

## Current accepted build

SHA256:1f52b6dffc744d3bc175beb53a90d49124e66ee264814aaf39fab766e92949a1
Backup:results/local/E186/candidate_unroll4.so. ABA shell trap restored it and
the package-local hash matched. No tmux experiment remains running at checkpoint.
Current precision scope:90%NVFP4/10%INT8 only, plus its denseINT8 capture stage.

Accepted source changes:
- Skip unused packedFP16 Q/K writes for no-pool/no-FP16 capture/reuse.
- Exact capture max reduces scores before shared positive scaling.
- Four warp-local half maxima, separately reduced to exact tile map. Q64 and
  Q128 each use4warps; earlierQ64 buffer bug was fixed and broad tests passed.
- Q128/D128/no-pool preparation compile-time specialization for NV+INT8 reuse
  and INT8-only capture, preserving generic fallback.
- E186 unrollfactor4 in Q/K preparation token loops only for specialized modes.

E186480p and768p124f full API output+route hashes equal original binary.
EarlierE176 checked Q64, alternate sparse budget and362fQ128, but finalE186
unroll has NOT yet been tested on long or later-step trajectory inputs.

## Final same-GPU ABA test

GPU1; real first-step block0 inputs;Q128,skip0,90/10;10warmups40timedcalls
perpath. OriginalA -> optimized -> originalB; original values below averageA/B.

| Input/path | Original mean ms | Optimized ms | Reduction |
|---|---:|---:|---:|
|480p124f capture|37.9368|33.0740|12.82%|
|480p124f reuse|29.6143|28.9856|2.12%|
|768p124f capture|140.0552|123.7479|11.64%|
|768p124f reuse|101.7949|101.1168|0.67%|

Controls: unchanged proxy path was0.65/0.95%slower in optimized round. There
was clock/system variation, so sub1%speedup is not a strong robust result.
Raw six reports:results/research/E186/aba. Output correctness hashes inE186.
Stage spans(E186initialrun):480p captureprep1.934ms/reuseprep3.944ms;
768p captureprep4.042/reuseprep7.831ms. Prefix overlaps; stage sums not additive.

## Attempt inventory

E171 full physical-route memoization:bitwise equal,~0.1ms480p,none768p;
not integrated. E174 warpQ16:bitwise equal, reuse41.7/146.8ms vs29/99;reverted.
E177doubleKbuffer:bitwise equal,~0.19ms480p only,none768p;reverted.
E180forced3CTA:bitwise equal but heavyspills,111.7/424.6ms;reverted.
E181Q64compute with originalQ128quantization/routes:bitwise equal but8–11%
slower;benchmark-only. E182/E183identityrescale skips:bitwise equal,40%fewer
nonfusedFP32instructions,but spillrequests1.71M->27.93M,no stablegain;reverted.
E184skip paddedqueryfragments:bitwise equal but~30/108ms,reverted. Firstcompile
had wrong helper argument location;corrected before actual test.
E185separate NV/INT8 stages withLSEmerge:Q128variant33.66ms vs28.92 at480p,
107.37ms vs~100ms at768p;additionalNMSE4.48e-6/2.74e-6,maxabs.089/.045.
Not integrated; experimental LSE host changes reverted.

## Main-kernel evidence and remaining work

NsightCompute at/usr/local/cuda-13.0/bin/ncu works. E179 baselineQ128mixed:
255regs/thread,16.7%occupancy,~2activewarps/scheduler,53%cycles noeligiblewarp,
DRAM2.4%,L2hit96.6%;fixed-latency and scalarFP32work prominent. Profiler locks
clocks,so its32msduration cannot be compared directly to normal22mskernel.
Route materializer actual isolated cost is tiny;in-chain3/16msspans were
queueing/contention and not removable arithmetic cost. Static90/10route has
almost allqueryrows using bothformats,limiting whole-representation pruning.

Need real same-layer step0->step10captures acrosslayers0,25,49(E187scripts
ready/notlaunched),then original/current pairedtiming with thosecachedroutes,
and final fullgeneration/longinputvalidation before complete claims.
