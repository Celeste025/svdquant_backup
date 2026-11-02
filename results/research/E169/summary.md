# E169 — H3 480p / 5s real QKV capture and route profile

## Capture and contract

Captured the first BF16 denoising call, block 0, race prompt 16, original
prompt/seed and CPU RNG. Output 480x832, 124 frames at 24 fps, 20-step H3
generation settings. The run stops at the requested attention call; it does
not generate a video. Q/K/V shape B,H,N,D = [1,56,15795,128] with BF16 data.
Video grid is [37,15,26] = 14,430 video tokens; contiguous non-video prefix
is 1,365 tokens. Capture:
`/data1/models/svdquant-wjq/research/20261101/E169_h3_480p5s_attention/race_480p_124f_step0_block0.pt`.

## Full API timings

GPU1 RTX PRO 5000 72GB, Q128, skip0, 90% NVFP4 + 10% INT8 mixed path,
INT8 prefix. Five warmups and 12 CUDA-event calls for each path:

| Path | Mean ms |
|---|---:|
| Proxy mixed each step | 29.867 |
| Plain INT8 first step | 33.115 |
| INT8 first step + exact importance capture | 36.809 |
| Cached exact-route reuse | 29.161 |

Capture adds 3.694 ms (11.15%) versus plain INT8 first step. Reuse saves
0.706 ms (2.36%) versus proxy mixed each step. For a 20-step single-layer
estimate, the capture premium breaks even after 6 reuse calls; 19 reuses
would save about 9.73 ms in total. This is not a measured DiT rollout gain.
Exact cached and proxy routes differ, so main attention is not identical work.

## Phase spans

| Phase (ms) | Proxy | Capture | Reuse |
|---|---:|---:|---:|
| Full API | 29.867 | 36.809 | 29.161 |
| QKV preparation | 5.247 (17.6%) | 3.396 (9.2%) | 4.811 (16.5%) |
| DraftMap score | 1.594 (5.3%) | 0 | 0 |
| Route selection | 0.408 (1.4%) | 0.168 (0.5%) | 0 |
| Route materialization event span | 0.020 (0.1%) | 0.054 compaction | 1.509 (5.2%) |
| Main attention | 22.015 (73.7%) | 33.000 dense+importance (89.7%) | 22.365 (76.7%) |
| Prefix attention, parallel | 2.134 | 2.214 | 2.119 |
| Output assembly | 0.435 | 0.467 | 0.436 |

Phase percentages are spans/full API, not an additive partition. Prefix
overlaps with the main stream and event spans can include queueing.

One-call CUDA kernel samples: mixed main attention is 22.042 ms proxy and
22.343 ms reuse; dense main is 26.850 ms ordinary and 30.996 ms with exact
importance. Packed-route materialization's own kernel sampled at 0.086 ms.
Repeated isolated full materialization calls averaged 0.026 ms by CUDA event
and 0.009 ms host-call time, much less than the 1.509 ms in-chain event span.

## Route memory

Current compact route cache: 2,585,860 bytes = 2.466 MiB per layer.
Materialized output including block ID and counts: 10,608,640 bytes =
10.117 MiB per layer. For 50 main layers, replacing the compact cache with
the full route would use 0.494 GiB, about 0.374 GiB more than current.

Raw reports: `q128_skip0_fp90_int10_124f.json`, `_phases.json`, and
`_kernels.json`. Sources: `scripts/research/capture_e169_h3_480p5s_qkv.py`,
`profile_e169_480p_paths.py`, and `validate_e169_480p_paths.py`.
