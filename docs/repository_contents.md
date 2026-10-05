# Repository contents and Git publication scope

This repository contains inference/quantization code and an exploratory research
record. It is not a release of a validated new quantization method. The current
working baseline is MiniMax-H3 with the existing SVDQuant recipe and official
SageAttention3. New idea research is paused; E084 completed the requested
historical Wan QAD timestep measurements.

## Start here

- [Research status](../research_state/00_state/current_state.md): authoritative
  current status; older report statements describe their historical stage.
- [Research index](../research_state/README.md): reports, experiments and decisions.
- [Intermediate overview](../research_state/reports/070_20261004_midterm_report.md).
- [Latest result: Wan QAD timestep replay](../research_state/reports/083_20261005_wan_qad_timestep_replay.md).
- [Research script guide](../scripts/research/README.md).

## Directory map

| Directory | Contents | Git policy |
|---|---|---|
| `nunchaku/`, `src/`, `app/`, `comfyui/` | Engine and integration source | Keep |
| `configs/`, `requirements/`, `scripts/`, `tools/` | Recipes, environments, inference and evaluation entrypoints | Keep source; exclude caches and compiled files |
| `third_party/` | Vendored patches and pinned submodules | Preserve existing source and submodule revisions |
| `research_state/` | Plans, literature checks, decisions, reports and figures | Keep; plans are not evidence of execution |
| `results/research/` | Experiment JSON/CSV, report figures, configuration and source snapshots | Keep allowed extensions; exclude private blind-review mappings |
| `results/mjvideo/`, selected `results/samples/` files | Existing evaluation records and manifests | Preserve existing allowlist |
| `results/logs/`, other run payloads | Local logs, media and tensors | Ignore |
| `outputs/` | Historical outputs | Retain existing text-record allowlist only |
| External data disk | Models, checkpoints, calibration states, captures and videos | Not part of Git |

Research evidence currently includes about 147 MiB of JSON, plus small tables,
figures and source snapshots. These are intentionally retained instead of silently
dropping detailed negative results. This is not a self-contained model bundle.

## Reading and reproducing experiments

Experiment IDs (`E022`, `E084`, etc.) and numbered report IDs are separate.
For example, report `083_...` documents experiment E084. Follow each report's
plan, input manifest, source hashes and output references, rather than choosing
the newest-looking script by filename. Failed attempts and superseded plans
are retained for provenance; their existence does not mean an experiment ran.

Many historical scripts and evidence records contain original server paths,
including `/home/wjq/workspace/svdquant-exp` and `/data1/models/svdquant-wjq`.
They are retained as recorded, not rewritten during repository cleanup. Absolute
links in older reports work on that server, not directly on GitHub. On another
machine, locate the matching repository-relative file, provision the external
artifacts and review path constants before running a launcher. A fresh clone
alone cannot reproduce experiments without the model/data assets and matching
CUDA/Python environments.

Do not automatically run historical launchers: they can initiate training,
generation, downloads or other resource-intensive jobs. The newest experiment
status and the user's current authorization determine what should run.

## Before committing and pushing

Inspect `git status`, `git diff`, and the exact staged diff. `.gitignore` does not
remove files already tracked. Review newly included result records for large
payloads and credentials. Keep weights, videos, tensor captures, logs, credentials
and `results/research/**/private/` local. Do not force-add those paths.

This cleanup changes ignore/documentation policy only; it does not relocate
artifacts, modify recorded experiments, create a commit, or push a remote branch.
