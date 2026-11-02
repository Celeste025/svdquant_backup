# E166 — detailed cost composition for proxy, exact capture and reuse

Sources: E163 four-path full API and phase data (5 warmups, 12 timed calls),
E164 one-call CUDA kernel profiles, E165 host-call spans and E166 isolated
packed-route materialization (20 timed calls). Q128, skip0, 90% NVFP4 / 10%
INT8, real MiniMax-H3 block-0 input, GPU2 RTX PRO 5000. The two inputs have
38,661 tokens/124 frames and 110,013 tokens/362 frames.

## 124 frames (ms; percent of corresponding full API)

| Path | Full API | QKV preparation | DraftMap | Route selection | Route materialization span | Main attention | Prefix (parallel) | Assembly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proxy | 100.57 | 10.61 (10.5%) | 3.38 (3.4%) | 1.02 (1.0%) | 0.05 (0.1%) | 84.89 (84.4%) | 4.4 (~4%) | 1.14 (1.1%) |
| Capture | 136.20 | 7.34 (5.4%) | 0 | 0.44 (0.3%) | 0.07 compaction (0.1%) | 127.83 dense+importance (93.9%) | 4.4 (~3%) | 1.14 (0.8%) |
| Reuse | 99.12 | 9.65 (9.7%) | 0 | 0 | 3.18 (3.2%) | 85.70 (86.5%) | 4.4 (~4%) | 1.13 (1.1%) |

## 362 frames (ms; percent of corresponding full API)

| Path | Full API | QKV preparation | DraftMap | Route selection | Route materialization span | Main attention | Prefix (parallel) | Assembly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Proxy | 757.34 | 30.16 (4.0%) | 17.04 (2.3%) | 4.59 (0.6%) | 0.47 (0.1%) | 703.13 (92.8%) | 18.9 (~2.5%) | 3.28 (0.4%) |
| Capture | 1086.63 | 20.76 (1.9%) | 0 | 4.00 (0.4%) | 0.31 compaction (0.0%) | 1058.74 dense+importance (97.4%) | 18.9 (~1.7%) | 3.24 (0.3%) |
| Reuse | 750.74 | 27.41 (3.7%) | 0 | 0 | 16.48 (2.2%) | 704.79 (93.9%) | 19.2 (~2.6%) | 3.28 (0.4%) |

These percentages are **per-phase span / full API**. They are not an additive
partition: prefix runs in parallel, and CUDA events around host calls can
include queueing. Stage sums exceed full API.

## Kernel-level and isolated findings

- The mixed main attention kernel dominates both normal and reuse paths:
  one-call kernel samples are 83.11/84.26 ms for proxy/reuse at 124 frames,
  and 687.64/690.87 ms at 362 frames. Different routes mean this is not a
  same-work kernel comparison.
- The dense importance kernel is 122.69 ms (124f) and 1022.14 ms (362f) in
  one-call kernel samples, compared with ordinary dense INT8 main kernel
  101.97 and 878.01 ms. It accounts for most capture overhead.
- Packed-route materialization's in-chain event span is 3.18/16.48 ms, but
  its isolated CUDA-event time is 0.059/0.412 ms. Its host call is about
  0.008 ms. The individual CUDA kernel sampled at 0.068/0.401 ms. The
  in-chain span is therefore not the materializer's compute cost; likely
  queueing or resource contention with overlapping prefix work. This remains
  a hypothesis until a stream timeline proves it.
- QKV preparation includes a mandatory input scan and quantization. Removing
  pooling saves about 0.97 ms (124f) and 2.78 ms (362f), based on same-input
  paired preparation calls. The remaining preparation still costs about
  9.6/27.4 ms in reuse.

## Implications

1. Improve or replace the dense importance capture kernel first; this is the
   dominant up-front penalty. Capturing the ordinary proxy route would avoid
   exact-importance work but changes the route semantics and needs validation.
2. Diagnose why the packed materializer sits in a 3/16 ms in-chain event span
   while its own kernel takes <0.5 ms. Do not optimize its arithmetic based
   on the span alone. Use an Nsight Systems stream timeline or equivalent.
3. A faster packed route alone cannot yield large end-to-end speedup while
   main attention consumes roughly 84–94% of later-step full API time. Test
   route changes against identical selected tile lists before attributing
   main-kernel differences to cache reuse.
