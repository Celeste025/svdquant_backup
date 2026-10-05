# Research script guide

These are experiment-specific programs, not a single supported CLI. Use the
[research index](../../research_state/README.md) and the matching experiment plan
to select an entrypoint. Historical variants remain in place because reports
record their paths and hashes.

| Group | Typical filenames | Role |
|---|---|---|
| Native quantization | `h3_native_nvfp4.py`, `wan_native_nvfp4.py`, `wan_nvfp4_fastpack.py` | Shared numerical/runtime implementation |
| QAD | `wan_mainweight_qad.py`, `train_wan_mainweight_qad_expanded.py` | Historical Wan main-weight training and native export |
| Experiments | `probe_*`, `run_*`, `bench_*`, `generate_*` | Specific interventions, timing or generation |
| Supervision | `launch_*`, `resume_*`, `supervise_*` | GPU selection, execution budgets and logs |
| Verification | `check_*`, `verify_*`, `summarize_*`, `evaluate_*` | Correctness checks, saved-result aggregation and evaluation |
| Native attention sources | Experiment-specific `.h`, `.cu`, `.patch` | Preserved implementations and patches |

## Latest completed follow-up: E084

- `launch_e084.py`: supervised replay of the immutable E022 test set on GPU0.
- `replay_wan_qad_timesteps_e084.py`: same-state predictions and free-trajectory
  replay; requires historical input/checkpoint artifacts and matching environment.
- `summarize_e084.py`: CPU verification and existing-video NMSE; `--plot-only`
  draws the saved timestep summary using matplotlib.
- [Plan](../../research_state/06_experiments/E084_wan_qad_timestep_replay_plan.md),
  [report](../../research_state/reports/083_20261005_wan_qad_timestep_replay.md),
  [results](../../results/research/E084/).

Replaying these commands is not necessary to read the published evidence.
Completed-run paths are intentionally protected against overwriting. Review
the plan and allocate a new output location before any intentional rerun.
