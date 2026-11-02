# E142 — real H3 attention precision/sparsity tradeoff (completed)

2026-10-31. Three true post-norm/RoPE BF16-model QKV captures replayed from
BF16 teacher trajectories. No new video generation or training.

| Sample | QKV HND shape | Prefix | Video grid |
|---|---|---:|---|
|Race step0 block0|1x56x22677x128|1365|37x18x32|
|Race step10 block25|1x56x22677x128|1365|37x18x32|
|Clap step10 block49|1x56x21731x128|419|37x18x32|

19arms per input: BF16 SDPA,Sage2++,optimized Sage3 and16 official Anemoi
configurations. Drop0/30/50/70/80% x retained INT8 share25/50/100%, plus
70skip/20FP4/10INT8. Fixed Q64,MeanMax0.2,official per-layer NVFP4 scales,
INT8 prefix queries/KV,0FP16 video phase. FP4 share is1 minus retained INT8
share. No all-FP4 Anemoi cell is included because fixed INT8 prefix-query
policy requires an active INT8 phase. Sage3 remains the all-FP4 dense baseline.

Full native API timed: centering,quantization,scoring/routing/packing,attention
and output assembly. Native contiguous HND/BSHD inputs prepared outside timing.
No model load/weight transfers/clones timed. CUDA events25warmups+6x12calls
each forward/reverse order:144 measured calls per arm.3same-model GPUs,
one per input; compare paired results within input before cross-GPU averages.

Sage2++ environment SM120 auto-dispatch is per-warp INT8 QK,E4M3 PV,
fp32+fp16 accumulation. Sage3 uses E130 readonly optimized native implementation,
actual110SM grid. Original/non-optimized Sage3 outputs exactly match optimized
outputs on all3captures, including large tail error; Q/K/V unchanged.
Native algorithms differ in centering/scales/quantization, not just precision.

All57arms measured complete output on all queries and56heads. JSON records
full/prefix/video MSE,NMSE and reference energy. Plot uses video-only NMSE
to prevent prefix from masking visual-token errors. No query sampling.

Representative results as latency ms / video NMSE percent:

| API |Race s0b0|Race s10b25|Clap s10b49|
|---|---|---|---|
|BF16|68.53 /0|70.61 /0|63.98 /0|
|Sage2++|38.16 /.0176|38.79 /.0210|35.27 /.1607|
|Sage3 optimized|29.07 /.4496|29.72 /.6689|27.23 /13.4268|
|50skip/25FP4/25INT8|27.82 /.3775|28.43 /.4387|24.46 /5.4765|
|50skip/50INT8|29.93 /.3465|30.55 /.3974|26.43 /1.9422|
|70skip/20FP4/10INT8|21.15 /1.3201|21.55 /.8511|17.66 /5.6670|
|70skip/30INT8|22.29 /1.2043|22.72 /.7376|18.69 /2.7377|
|80skip/20INT8|18.43 /2.7520|18.79 /1.2425|14.82 /3.3221|

Takeaways within checked calls:

-50/25/25 beats optimized Sage3 on latency and video NMSE on all3calls.
-At Sage3 latency budget,the lowest-error tested Draft cell is50/25/25
 for both race calls,but50skip/50INT8 for clap block49.
-70skip/30INT8 loses about1.0–1.2ms to70/20/10 but has lower NMSE on all3calls.
-Sage2++ is slower but much lower error on these captures.
-Tail block49 is strongly FP4-sensitive; do not hide it in an average.
 Its FP4-mixed error can decrease with greater sparsity, unlike the race calls.
 This observation does not isolate quantization-error cancellation or prove
 a new routing method; precision allocation and normalization also change.

These are selected3calls,not a model-wide or video-quality guarantee. E141
already shows single-operator and closed-loop video ordering can differ.
Use identified cells as generation candidates,not final deployment recipes.

Artifacts: tradeoff.png (linear),tradeoff_log.png (log-y),tradeoff.csv (57rows),
summary.json (per-input budgets),three raw timing/error JSONs and
sage3_validation.json. QKV2.7GiB under
/data1/models/svdquant-wjq/research/20261031/E142_attention_tradeoff.
First capture attempt failed on hook keyword naming before producing tensors;
failure logs retained. Plot exported using existing PTQ matplotlib environment.
All generation/capture/benchmark/validation processes ended.
