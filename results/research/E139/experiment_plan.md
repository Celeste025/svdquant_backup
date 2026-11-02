# E139 — Official DraftAttention2 H3 generation screen

User requested official code and generation before routing optimization.
Source: https://github.com/anemoi-project/anemoi
Pinned commit: 97a1c8b14f26f387214c8cfb8fc7e890390b93b4
Checkout: third_party/anemoi-code (filtered clone, no display media).

Reuse H3 BF16 model, scheduler, race/clap prompt/seed and 576x1024/124-frame,
20-step settings from E121/E125. Keep linear layers BF16. Use official
anemoi_attention implementation and official H3 layer calibration when available.
Respect packed non-video prefix, video THW, separate padding segment, and
original BF16 refiner. Do not classify non-video prefix as spatial video.

Primary requested arm: drop 70% video-region pairs; among retained pairs,
NVFP4=2/3 and INT8=1/3 (20% and 10% of all video-region pairs).
Control: same 70% sparsity, retained INT8=1.0. Reuse matching BF16 references.
Default official Mean/Max scoring stays unchanged. Dense-first policy must be
recorded explicitly; nominal ratios apply to sparse video blocks only, not
prefix or dense calls. No inference claim based only on nominal bit percentages.

Preflight: compile official SM120 extensions; small dense FP16 numerical check,
real H3 QKV shape/layout/finite-output smoke. Confirm actual prefix and video
counts and preserve inverse spatial token order. Then run two prompts and two
new attention arms in named tmux sessions on actually idle GPUs.
Report all-step timing (exclude model load), source/config/initial-latent hashes,
video NMSE and LPIPS to matched BF16, plus playable original videos. Few-case
reference distances do not establish a broad perceptual-quality ranking.

Data: /data1/models/svdquant-wjq/research/20261031/E139_draftattention2
Build/cache: /data1/models/svdquant-wjq/research/20261031/E139_draftattention2_build
No existing jobs/checkpoints are overwritten. No routing innovations in this run.
