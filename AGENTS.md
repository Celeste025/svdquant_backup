# Agent handoff guide

## Workspace and version control

- **Repository:** `/home/wjq/workspace/svdquant-exp`
- **Branch:** `main`, tracking `origin/main` (`https://github.com/Celeste025/svdquant_backup.git`)
- This directory was restored from a ZIP checkout on 2026-09-14. Do not assume
  old absolute paths in archived scripts are valid; prefer the paths below.
- Do not push without the user's explicit instruction. Before committing, review
  `git diff --cached --stat` and preserve existing files unless removal is deliberate.

## Runtime locations

| Purpose | Path |
|---|---|
| Shared models, checkpoints, calibration caches | `/data1/models/svdquant-wjq` |
| Historical DeepCompressor/PTQ Python | `/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python` |
| MJ-VIDEO Python | `/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python` |
| DiffSynth checkout | `/home/wjq/workspace/DiffSynth-Studio` |
| VBench metadata | `/home/wjq/workspace/ViDiT-Q/eval/video/Vbench/vbench/VBench_full_info.json` |
| MJ-VIDEO source/model | `/data1/models/svdquant-wjq/third_party/MJ-Video`, `/data1/models/svdquant-wjq/models/MJ-VIDEO-2B` |

Models, caches, generated media, and tensor checkpoints are intentionally not
versioned. Put reusable source in `scripts/` or `configs/`; put reviewable
tables and JSON reports under `results/`.

## Code layout

- `scripts/`: experiment entrypoints, inference workers, validation, and report
  export helpers. Prefer an existing script over inventing a launch command.
- `third_party/deepcompressor/`: vendored and locally patched PTQ runtime.
  Treat its modifications as project code; do not overwrite it from upstream.
- `configs/`: small project-local experiment configurations.
- `results/`: generated artifacts. Git retains MJ-VIDEO outputs, sample
  manifests, and per-case `mjvideo_metrics.json`; broad intermediate reports,
  videos, and tensor artifacts remain local by default.

## Current rCM-Wan setup

- Base: `Wan2.1-T2V-1.3B-Diffusers` with rCM transformer at
  `models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer`.
- Real-NVFP4 SVDQuant checkpoint:
  `ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16` — W4A4 E2M1, group 16, rank-32
  low-rank branch, layerwise OutputsError smoothing.
- Paired INT4 SVDQuant checkpoint:
  `ckpts/rcm-wan2.1-1.3b-int4-s16-g10` — dynamic W4A4 signed INT4, group 64,
  rank 32, ten smoothing grids. Its PTQ recipe is
  `scripts/run_rcm_wan_int4_s16_g10.sh` and
  `third_party/deepcompressor/examples/diffusion/configs/svdquant/rcm_wan_int4_s16_g10.yaml`.
- rCM VBench contract: 480x832, 77 frames, 4 denoising steps, `sigma_max=80`,
  guidance 0, and 16 FPS. See `scripts/rcm_vbench251_worker.py` and
  `scripts/run_rcm_int4_vbench51.py`.

## Current MiniMax-H3 setup

- Common runtime/helper: `scripts/minimax_h3_svdquant_common.py`.
- Standard SVDQuant inference: `scripts/infer_minimax_h3_svdquant_standard.py`.
- Prompt table: `/home/wjq/workspace/178866172854036` (JSONL, not tracked).
  Always recover prompt text from `prompt_id`; do not infer it from filenames.
- Standard paired samples: `results/samples/minimax_h3_svdquant_standard_8p64s`.

## Evaluation and long-running jobs

- MJ-VIDEO evaluator: `scripts/eval_mjvideo_rcm_vbench51.py`; report exporter:
  `scripts/export_mjvideo_report.py`. The evaluator supports either a VBench
  manifest or `--prompt-source` plus `--prompt-ids`.
- MJ-VIDEO review tables use total, alignment, fineness, and coherence. Safety
  and bias/fairness remain in raw output but are excluded from review tables.
- Run expensive generation/evaluation inside a named `tmux` session, log to
  `results/logs/`, and verify GPU availability with `nvidia-smi` before launch.
