# Optimization progress through E176

Goal remains active; reuse/main-attention deep optimization is incomplete.

Current package binary SHA256:
92cf7a20583f9fd4c91d1f0eaf278e2357136ce681d79001655855d2dc5dcee6.
Identical accepted backup: results/local/E176/accepted_e173.so.

## Accepted E173 capture change

Each warp writes its own scaled half tile maximum without CTA barriers.
The host allocates [B,H,Qblocks,Kblocks,4], launches dense INT8 attention,
then reduces the final warp dimension with amax. Both Q64 capture (warp-Q16)
and Q128 capture (warp-Q32) have four warps. Initial QueryBlock/32 allocation
was wrong for Q64 and caused an illegal access in the first broad test. Fixed
to match the four-warp instantiations, rebuilt, and reran all checks.

Output and captured-route hashes are bitwise equal to the original E154
binary for real 480p/124f Q128, 768p/124f Q128, 480p Q64 skip75/FP60,
768p Q64 skip0/FP90 and 768p/362f Q128 skip0/FP90 inputs. E176 retry log
ends with ALL_VARIANTS_BITWISE_EQUAL; binary restoration hashes match.

Capture full API, GPU1, 5 warmups/12 calls:
480p E172 35.914 -> E173 33.266 ms;
768p E172 134.171 -> E173 122.436 ms.
Reuse remains ~28.997 / 99.120 ms. Long performance run is now tmux
e176_long_perf, log results/logs/e176_long_perf.log and report
results/research/E176/long_perf.json. Poll its actual handle before any restart.

## Rejected E174 warp layout

Changed only Q128 mixed NVFP4/INT8 instantiation MPA_WARP_Q32 to16.
Bitwise identical output, but reuse regressed 28.997 ->41.724 ms at480p and
99.120 ->146.815 ms at768p. Registers increased; old source and binary
were restored. Candidate binary retained as results/local/E174/rejected_warp16.so.

## E175 static-route preparation limit

At480p, 8288 query rows all use FP4; only28 rows have zero INT8 tiles.
At768p, 16576 rows all use FP4; only8 rows have zero INT8 tiles.
Thus eliminating whole Q precision representations by static row masks has
little upside for the current90/10 policy. Dense prefix also needs INT8 K/V.
Data in E175/*_hashes.json (contains activity statistics).

## Next main-attention experiment

NVFP4 loop in q64_attention_phase_composer.inl already overlaps current V
load with QK, next K with PV, but uses one K buffer and a CTA barrier afterQK
before overwriting it. Test two K buffers with alternating iteration parity:
issue next K into the other buffer after currentQK and before softmax,
remove that overwrite-protection barrier, preserve V-ready and final wait
barriers. Existing shared-memory reservation for mixed INT8 (~33KB) appears
large enough for NV buffers (~13.5KB), but check all HeadDim/phase variants
and actual offsets before editing. Measure full API and exact output hashes;
do not infer speed from fewer barriers alone.
