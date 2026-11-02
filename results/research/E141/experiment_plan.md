# E141 — first2 steps dense, no layer exemption

User requested10% dense warmup in20-step H3 generation. Set
dense_first_steps=2 and dense_first_layers=0. Steps0/1 all50 main attention
blocks BF16 dense; steps2–19 all50 blocks use DraftAttention2 (900 sparse calls).
Per-step routing assertions remain active. Keep BF16 linears, Q64,70% dropped
video-region pairs, MeanMax0.2, official per-layer scales and INT8 prefix.
Refiner/independent padding retain original helpers.

Race/clap x two arms: retained30% INT8, or20% NVFP4+10% INT8. Same checkpoint,
prompt,seed,initial noise,576x1024,124frames,20steps,CFG1,flowshift12/3 as E139/E140.
Reference BF16 race E121, clap E139. Compare E139 (10steps+2layers dense),
E140 (no dense exemption), E141 (2steps only), without claiming this isolates
the separate step/layer effects of E139.

Artifacts: four videos, full-video NMSE/LPIPS, per-step timing, schedule/source
manifests and two four-column videos (BF16,all-sparse mixed,first2 INT8,first2 mixed).
Data: /data1/models/svdquant-wjq/research/20261031/E141_draft_first2_dense.
