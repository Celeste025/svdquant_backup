# E157 — Q128 later-step route reuse phase profile

Representative MiniMax-H3 block-0 capture: 38,661 tokens, 56 heads, D128.
GPU 2: RTX PRO 5000 72GB; no other compute process observed at launch or completion.
One warmup and three measured calls per path. Full API timings are uninstrumented;
phase spans are measured in separate instrumented calls. Prefix attention overlaps.
Phase times include operator dispatch and allocation gaps; they are not individual
CUDA kernel durations. This is a small smoke, not an exhaustive benchmark.

| Phase (ms) | Skip75 proxy | Skip75 reuse | Skip0 proxy | Skip0 reuse |
|---|---:|---:|---:|---:|
| Full API |42.075|40.676|99.410|98.098|
| QKV preparation, optional pooling included |10.618|9.655|10.638|9.639|
| DraftMap score |3.214|0|3.316|0|
| Precision route selection |0.991|0|1.002|0|
| Route materialization |0.036|3.007|0.055|3.135|
| Attention main |25.907|26.861|83.338|84.438|
| Prefix, overlapping |4.214|4.234|4.343|4.335|
| Output assembly |1.142|1.125|1.136|1.126|

Paired replay of identical preparation inputs with pooling on/off:
- Skip75: 10.605 / 9.628 ms, incremental pooling 0.977 ms.
- Skip0: 10.592 / 9.619 ms, incremental pooling 0.973 ms.

The removed pooling + scoring + selection amounts to about 5.2–5.3 ms.
The packed-cache materialization stage adds about 3 ms relative to legacy,
and the different exact-max route has about 1 ms slower main attention in this
smoke. These effects explain why total savings are only 1.3–1.4 ms here.
Exact-max and legacy proxy routes differ: this is not a same-route isolation.
The reason packed materialization is slower remains to be established with
kernel/host-level profiling; source inspection alone does not prove the cause.
The operator expands packed int16 IDs into a full-width int32 physical list,
including clearing the output and inserting prefix entries, every reuse call.

Artifacts: *_phases.json, full reports, scripts/research/profile_e157_route_reuse_phases.py,
results/logs/e157_gpu2.csv, results/logs/e157_phases.log.
