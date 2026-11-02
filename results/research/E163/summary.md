# E163 — proxy, exact-route capture and reuse profile

## Scope

One captured MiniMax-H3 block-0 Q/K/V input per length, Q128, 56 heads, D128,
zero skipped tiles, 90% NVFP4 / 10% INT8 on mixed calls, INT8 text prefix.
The 124-frame and 362-frame captures have 38,661 and 110,013 tokens.
RTX PRO 5000 72GB (GPU2). Full-API CUDA events: five warmups and twelve
measured calls per path. Phase spans were measured in separate instrumented
calls. They include dispatch/allocation gaps and overlapping prefix work, so
their sums are not critical-path latency or isolated kernel durations.
Same captured input is replayed for all paths; these are operator calls, not
full DiT or generation timings.

## Full API (ms)

| Frames | Proxy mixed each step | Plain INT8 first step | INT8 first step + exact capture | Cached route reuse |
|---:|---:|---:|---:|---:|
| 124 | 100.565 | 119.658 | 136.198 | 99.123 |
| 362 | 757.343 | 944.377 | 1086.633 | 750.745 |

Compared with ordinary proxy mixed, reuse saves 1.443 ms (1.43%) at 124
frames and 6.598 ms (0.87%) at 362 frames. Compared with the old plain INT8
first-step call, exact capture adds 16.540 ms (13.82%) and 142.256 ms
(15.06%), respectively. The first step's ordinary INT8 API itself still
runs DraftMap scoring and route selection despite zero skip.

For a 20-step sequence with one captured first call and 19 reuse calls, the
single-layer arithmetic break-even is 12 reuse calls at 124 frames and 22 at
362 frames. Under these timings, 20 steps yield about 10.87 ms net saving for
124 frames and 16.89 ms net loss for 362 frames. This is a timing estimate,
not a measured end-to-end DiT saving. The exact cached route differs from the
proxy route, so output and main-kernel work are not identical.

## Main phase spans (ms)

| Frames | Path | QKV prepare | DraftMap | Route select | Route materialize | Main attention |
|---:|---|---:|---:|---:|---:|---:|
| 124 | Proxy mixed | 10.61 | 3.38 | 1.02 | 0.05 | 84.89 |
| 124 | Reuse | 9.65 | 0 | 0 | 3.18 | 85.70 |
| 124 | Plain INT8 | 7.96 | 3.19 | 1.16 | 0.06 | 106.79 |
| 124 | Capture | 7.34 | 0 | 0.44 | 0.07 compaction | 127.83 dense+importance |
| 362 | Proxy mixed | 30.16 | 17.04 | 4.59 | 0.47 | 703.13 |
| 362 | Reuse | 27.41 | 0 | 0 | 16.48 | 704.79 |
| 362 | Plain INT8 | 22.56 | 16.65 | 4.65 | 0.47 | 904.74 |
| 362 | Capture | 20.76 | 0 | 4.00 | 0.31 compaction | 1058.74 dense+importance |

The later-step saving from skipping pooling, DraftMap and selection is largely
consumed by the *in-chain event span* around packed-cache materialization.
That span goes from about 0.05 to 3.18 ms (124f), and 0.47 to 16.48 ms
(362f). Later E164/E166 measurements show the packed CUDA kernel itself takes
only about 0.06/0.41 ms in isolation, and the host call about 0.008 ms.
Thus the in-chain span includes queueing/resource contention or other
stream-timeline effects; it must not be labeled as the materializer kernel's
compute time. The capture bottleneck is the dense INT8 attention kernel with
in-kernel exact importance collection.

## Next implementation targets

1. Profile the packed-route materializer at kernel level, then avoid expanding
   the int16 cached IDs to a full-width int32 physical list on every reuse if
   the attention kernel can consume a compact representation.
2. Profile the dense importance kernel at kernel level, especially its tile-max
   reduction/synchronization. Consider capturing the proxy route from ordinary
   first-step scoring as an alternative design; it changes route semantics and
   needs output validation.
3. Repeat full-API measurements on multiple real blocks/steps before claiming
   a model-level speedup. The present result uses one block-0 input per length.

Raw reports: `q128_skip0_fp90_int10_{124,362}f_four_path.json` and matching
`_phases.json`. Source: `scripts/research/profile_e163_three_paths.py` and
`scripts/research/validate_e163_four_paths.py`.
