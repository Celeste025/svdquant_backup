# E220: concurrent old-route output and probability capture

The refresh-step contract is: step 8 uses the route captured at step 2 for mixed output and captures a new map for step 9 onwards; step 14 uses the route captured at step 8 and captures a new map for step 15 onwards. Step 2 is dense INT8 with capture. This experiment does not launch the four-video third arm.

On GPU 5 RTX PRO 5000 72GB, E187 race 480x832 124-frame step0 layer25 QKV, Q64, all 56 heads, mixed 10% INT8/90% NVFP4 and no skip: 2 warmups, 8 CUDA event samples. Mixed-only 25.53 ms; full fused dense INT8 capture 34.48 ms; simultaneous separate-stream calls with a join 60.35 ms, +136.38% vs mixed. Mixed output NMSE 0; captured block IDs and counts match independently captured routes for identical QKV. This measures one real operator input, not E202 768p video or end-to-end speed; samples in overlap.json. Concurrent full APIs duplicate operand preparation, dense output and QK/PV, and compete for the same GPU. It does not establish a lower bound on a new fused kernel.

Current E219 third-arm code implements correct old-route-before-refresh semantics sequentially, as a correctness prototype only. An optimized implementation is not available, and <20% is not demonstrated. Required next work: design shared preparation and/or kernel-level selective mixed-output plus full INT8 probability statistics; validate old-route output and full probability-map routing independently on actual different-step QKV, then benchmark both video lengths at 768p before any rollout.

Entry: scripts/research/bench_e220_overlap_capture.py; launch: scripts/research/launch_e220_overlap.sh; logs: results/logs/E220_overlap_check.log.
