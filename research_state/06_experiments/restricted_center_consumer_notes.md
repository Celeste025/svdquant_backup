# Restricted-center correction consumer: implementation boundary

Status: read-only source analysis; no kernel implementation, compilation, or GPU run for this note. E029's sample-adaptive rank16 signal motivates checking feasibility, not a speed claim. E030 has added a separate `factorable_fp32` precision arm (three layers, six probes total); results were not available when this note was written.

## Proposed interface and reusable path

For each head and Q block, use `c = global + A B`, with rank16 `B`, and precompute `T = [global; B] Kc^T` (17 rows). New inputs would be `A[head,Qblock,16]` and `T[head,17,K]`, with an explicit factorized mode and matching strides. The current pointer/layout represents already combined correction rows; it cannot silently be reinterpreted as 17 basis rows.

The existing SM120 launch is M128/N128/D128 with three stages. DS has zero stride in M, so the consumer currently receives **one 128-float correction vector per K tile**, broadcast over 128 query rows; the global allocation is compact `[B,H,Qblocks,K]`, not a full `[B,H,Q,K]` score tensor. The two N64 score slots are seeded at `add_delta_s_slot`, then the original FP4 QK MMA accumulates into them. This is the appropriate numerical insertion point: preserve its slot lifecycle and the subsequent P/PV/softmax logic.

Sources (absolute local paths and one-based lines):
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/api/launcher.h:129` — launch traits and three stages.
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/kernel/traits.h:164` — zero-stride DS layout.
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/mainloop.cuh:196` — existing DS descriptor construction; `:648` — `add_delta_s_slot`; `:817` — slot reuse; `:859` — initial score seeding.

## Plausible implementation: producer combines, consumer stays compact

The Mainloop producer warp can cooperatively compute the correction: 32 lanes, four key columns per lane, load 17 rows from global memory, accumulate `T0[n] + sum_r A[m,r] T[r,n]`, and write the existing 128-float DS stage. This uses 2048 unique FP32 FMAs, 8.5 KiB of source values, and a 512 B result per K tile. Cache reuse across Q blocks is possible, not guaranteed. No promise of L2 residency or latency hiding follows from the smaller global table.

This avoids both naive consumer duplication (the current lane-to-column mapping would recompute each column about 64 times if the dot product were inserted independently in every consumer thread) and staging all 17 rows. Three stages of FP32 17-row inputs require 25.5 KiB versus the current 1.5 KiB DS storage, an extra 24 KiB. Keeping the other current arrays, even Q/K/V/O plus this new DS storage alone is at least 116224 bytes, above the local CUTLASS SM120 capacity constant of 101376 bytes; scale arrays and alignment add more. Thus a full three-stage 17-row TMA replacement does not fit the unchanged storage layout.

A can be distributed among lanes and broadcast with shuffles rather than replicated as 16 live coefficients per thread. The current producer warpgroup releases registers down to 24 per thread; consumers request 232 in the noncausal path. Four accumulators plus streamed values and addressing still need compilation to establish actual registers/spills. Producer LDG/FMA work also competes with its job of issuing Q/K/V TMA, so eliminating correction materialization need not improve latency.

Sources:
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/kernel/traits.h:39` — separate shared arrays; `:80` — 12-warps configuration.
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/cutlass/include/cutlass/arch/arch.h:45` — SM120 shared-memory capacity constant.
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/kernel/attention_kernel.h:52` — consumer register allocation; `:154` — producer allocation; `:158` — Mainloop producer role; `:182` — separate Epilogue producer role.

## Synchronization is a real change

The current hot loading loop is inside `elect_one_sync()`'s single-lane branch. K, its scales, and DS are issued against the same K pipeline barrier. Removing the DS TMA requires subtracting its bytes from the K expected transaction count, but that alone is insufficient: CUTLASS `producer_acquire` already performs `arrive_and_expect_tx`. A K/SF TMA completion must not release consumers before cooperative ordinary DS stores are published.

A legal design would separate waiting for the empty stage from publishing the full stage: wait until old consumers release it; cooperatively calculate/store DS; gather warp writes with warp synchronization; then have the leader perform the release arrival/expectation for K+SF and issue those TMA operations. Consumers retain the acquire wait before reading DS. Overlapping DS computation with K/SF TMA instead requires an additional DS-ready arrival condition or equivalent explicit protocol. A warp sync or standalone fence is not, by itself, the cross-warp completion condition. DS is then read by ordinary shared loads; any async-proxy fence requirement must follow the final accesses/layout, not be assumed to replace release/acquire synchronization.

Sources:
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/mainloop.cuh:131` — K transaction bytes include DS; `:451` — elected-lane prologue; `:483` — elected-lane steady loop.
- `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:512` — acquire waits empty, then arrives/expects transactions; `:560` — ordinary TMA `producer_commit(bytes)` is a production no-op, so it is not an implicit DS-store commit.

## Precision and decision boundary

E029 used `c = BF16(global + A B)` for both Q centering and `c.float() @ Kc.float().T`. Elementwise BF16 rounding can break exact rank17. A factorized consumer of the unrounded `global + A B` would differ by `[BF16(global + A B) - (global + A B)] Kc^T`, in addition to arithmetic-order differences. E030's separate `factorable_fp32` arm explicitly addresses this representation mismatch; no result is asserted here.

Source: `/home/wjq/workspace/svdquant-exp/scripts/research/probe_h3_restricted_centers.py:118` — rounded center and matched correction; `:168` — documented rank-breaking rounding boundary.

The producer design is feasible in principle and preserves most consumer logic, but remains unimplemented: barrier ordering, actual register use, cache behavior, total preprocessing plus attention latency, and fixed-basis transfer across samples are unmeasured. Neither low-rank algebra nor a smaller correction allocation establishes novelty or a deployment benefit. The relevant performance comparison remains the measured whole path and Q-chunking baseline, not the 177-to-17 storage ratio.

## One-hot codebook alternative: narrower interface, still unimplemented

E031 is complete; E032 has frozen a K16 codebook with six Lloyd iterations, with no kernel implementation or result asserted here. For `c_i=C[id_i]`, precompute `T=C Kc^T` and use the same actual C values for Q centering and correction. Each CTA loads one center index and selects its T row. Unlike dense rank16 coefficients, no 17-row combination is needed: each KV tile still transfers 128 FP32 values (512 B) into the existing compact DS stage. Existing DS shared storage, K-pipeline transaction bytes, TMA completion protocol, and consumer score seeding can remain unchanged in this design; it does not require the cooperative producer-store protocol described above for dense A.

The concrete change is the global DS row selected in `/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/attention/sm120/nvfp4_attention_sm120/compute/mainloop.cuh:412`: replace the current Qblock-dependent row with `center_id[batch,head,Qblock]`. Descriptor construction at `:196`, argument plumbing and host validation need a separate codebook-row extent; the true Q sequence length must continue to control Q access, scheduling, masking, and output. Merely substituting a shorter DS allocation under the old descriptor would be unsafe. Selecting the same C member on both paths also avoids the PCA-specific issue where elementwise BF16 rounding destroys an exact low-rank factorization. Normal matmul/reduction rounding remains.

This is an interface feasibility argument, not compiled code or a speed result. There is still clustering/assignment and T-construction cost, and shared row reuse may affect cache behavior. The family is more restrictive than continuous coefficients, so mean reconstruction or native accuracy cannot be assumed to match rank16 PCA.

Prior-work boundary: [Clustered Attention §3.2–3.3](https://arxiv.org/pdf/2007.04825) already clusters queries and shares centroid-based attention, with top-k refinement, but approximates the attention distribution/output rather than retaining the full low-bit Q-minus-center product over all keys. [VC-Attention §3.2 and Appendix B](https://arxiv.org/html/2609.15810v1#S3.SS2) clusters V, permutes KV, and restores fixed hardware-block V means; it is not this Q-block codebook interface. The limited check found no complete equivalent FP4 implementation, which is not a novelty proof. This remains a classical strong baseline; no new method claim follows from one-hot lookup.
