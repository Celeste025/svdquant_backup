# E140 — all-step/all-layer DraftAttention2 (completed 2026-10-31)

User requested removal of E139's dense warmup. Same official Anemoi backend,
Q64,70% video-region sparsity,MeanMax0.2,official H3 layer calibration and
INT8 non-video prefix. All200 target linear layers BF16. No kernel changes.
Set dense_first_steps=0 and dense_first_layers=0: all20 steps x50 main blocks
use the configured sparse path. Each step asserts exactly50 sparse calls.
Refiner and independent padding remain original helpers. Prefix interactions
remain retained under the official API; all-sparse refers to main video blocks.

Same race/clap prompt/seed,576x1024,124frames,20steps,CFG1,flow shifts12/3.
Initial video/audio hashes match references (race E121, clap same-environment
E139 BF16). Four arms completed20 steps with finite outputs and124-frame
audio/video media. Both exported comparison videos also contain124 frames.

| Case | Schedule / precision | RGB NMSE vs BF16 | LPIPS-Alex | Total DiT seconds |
|---|---|---:|---:|---:|
| Race | E139 warmup, mixed | 0.019715 | 0.106335 | 187.84 |
| Race | All sparse,30% INT8 | 0.209697 | 0.352563 | 172.74 |
| Race | All sparse,20% FP4+10% INT8 | 0.280485 | 0.412350 | 175.39 |
| Clap | E139 warmup, mixed | 0.021480 | 0.067990 | 188.16 |
| Clap | All sparse,30% INT8 | 0.557856 | 0.450466 | 167.84 |
| Clap | All sparse,20% FP4+10% INT8 | 0.548970 | 0.443323 | 174.33 |

Lower metrics mean closer to matched BF16, not a perceptual quality ordering.
Mixed has larger distances than INT8 on race but slightly smaller on clap.
Removing both exemptions changes composition/viewpoint and hand placement
substantially in sampled frames; it does not by itself isolate which early
steps or layers matter. Single rollout timing includes weight management and
is not a repeated native-attention benchmark. Mixed total DiT time falls
6.6%/7.4% versus warmup; no stable mixed-over-INT8 timing benefit is established.

Comparison video columns:
BF16 | E139 mixed with dense-first10steps/2layers | all-sparse INT8 | all-sparse mixed.
They are silent; individual original videos retain audio. Contact frames25,
63 and101 were inspected; no full temporal-quality claim follows from those
stills. Per-case JSON records metrics, initial hashes, timings and source hashes.

Media root: /data1/models/svdquant-wjq/research/20261031/E140_draft_no_dense.
Sources: run_e140_draft_no_dense.py, configurable run_e139_draftattention2.py,
finish_e140_draft_no_dense.py under scripts/research. E139 original outputs are
preserved. All generation/evaluation jobs completed, no idle-GPU watcher.
