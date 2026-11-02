# E141 — first2 steps dense (completed 2026-10-31)

Set dense_first_steps=2,dense_first_layers=0. Steps0–1 dense BF16 main
attention; steps2–19 all50 main blocks sparse (900 sparse calls per arm).
Same E139/E140 Q64,70% skip,MeanMax0.2,official H3 calibration,INT8 prefix,
BF16 linears,original refiner/padding. Same race/clap prompts,seeds,initial
noise,576x1024,124frames,20steps,CFG1,flowshifts12/3. No kernel/scoring change.

Four finite20step rollouts and124-frame videos completed. Every step asserts
the expected sparse-call count; initial video/audio hashes match BF16.
Both124-frame four-column exports validated. All jobs ended.

| Case | Retained precision | RGB NMSE | LPIPS | Total DiT seconds |
|---|---|---:|---:|---:|
| Race |30% INT8|0.173415|0.306524|172.59|
| Race |20% FP4+10% INT8|0.172909|0.315638|185.09|
| Clap |30% INT8|0.408235|0.351984|167.16|
| Clap |20% FP4+10% INT8|0.126528|0.195638|172.57|

Mixed schedule comparison:

| Case | Dense exemption | RGB NMSE | LPIPS |
|---|---|---:|---:|
|Race|None E140|0.280485|0.412350|
|Race|First2steps E141|0.172909|0.315638|
|Race|First10steps+first2layers E139|0.019715|0.106335|
|Clap|None E140|0.548970|0.443323|
|Clap|First2steps E141|0.126528|0.195638|
|Clap|First10steps+first2layers E139|0.021480|0.067990|

Lower distances mean closer to matched BF16, not a perceptual-quality rank.
First2steps improve both mixed cases, especially clap; sampled frame63 shows
viewpoint/background closer to BF16. One seed per case, no preference study.
Precision ordering is case-dependent. E139 also protects2layers, so these
schedules do not isolate warmup length alone. Timing is single-run DiT with
weight management; do not infer stable speed ordering from these totals.

Video columns: BF16 | E140 all-sparse mixed | E141 first2dense INT8 |
E141 first2dense mixed. Silent comparisons; originals retain audio.
Media root: /data1/models/svdquant-wjq/research/20261031/E141_draft_first2_dense.
Per-case video_metrics.json contains sources,hashes,metrics and step timings.
