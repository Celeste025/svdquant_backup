# Experiment log

## E085 — started 2026-10-06

MiniMax-H3 SVDQuant timestep NMSE split validation.  The calibration-side
teacher-forced measurement already exists at
`results/reports/minimax_h3_svdquant_standard_8p64s/calib_teacher_forced_step_error.json`.
The held-out side reuses standard sample prompt IDs 2, 16, 26, 42, and 51.
Prompt 16 has an existing full teacher-forced output; prompts 2, 26, 42, and
51 are to be run on GPU 1.  See `decision_value_plan_E085_h3_svdquant_timestep_generalization.md`.
