# E140 — remove both dense-first exemptions

User requested all20 steps and all50 main DiT blocks use DraftAttention2.
Set dense_first_steps=0 and dense_first_layers=0. Keep E139 code/backend,
Q64, MeanMax weight0.2, calibration, prefix INT8 and BF16 linears unchanged.
Every step asserts exactly50 sparse main-attention calls (1000 per rollout).
Refiner and independent padding helper retain the original pipeline behavior;
all prefix interactions remain retained under the official prefix contract.

Race and clap: same prompt/seed/initial noise/scheduler,576x1024,124frames,
20steps, CFG1. Two arms:70skip/30INT8 and70skip/20NVFP4/10INT8.
Reuse race E121 BF16 and clap E139 same-environment BF16. Also compare
E139 warmup videos, full-video NMSE/LPIPS and DiT timings. Four new videos,
same original BF16 linear model; no new weights/calibration or kernel changes.

Data: /data1/models/svdquant-wjq/research/20261031/E140_draft_no_dense.
Reports preserve per-arm manifests and first-step per-layer path records.
