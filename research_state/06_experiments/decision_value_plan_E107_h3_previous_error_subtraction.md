# E107 — previous-step oracle error subtraction

- Experiment ID: E107.
- Mode: exploration / mechanism diagnostic.
- Problem addressed: E103/E104 measured adjacent-step signed output-error alignment. Does subtracting a fixed fraction of the preceding true error reduce the next step's paired output NMSE?
- Hypothesis: On the saved BF16 trajectory, corrected error `e_t - k e_{t-1}` has lower NMSE than `e_t` for `k=0.5` and/or `k=0.8`, where `e_t = Q_t-BF16_t`; `k=0.8` need not always beat `k=0.5` because the error scale changes with timestep.
- Decision informed: Whether temporal error predictability is large enough to justify studying a deployable correction estimator or trajectory-aware calibration objective.
- Minimum valid setup: Use all 20 hash-registered E103 race and E104 clap paired raw calls, aligned by step and complete-DiT video/audio tensor coordinate. Do not mix video with audio, or use independent rollout outputs as if aligned. Evaluate steps 1–19 only, both fixed coefficients, per-step NMSE with the *next/current step BF16 output energy* denominator, and pooled NMSE with pooled numerator/denominator. Record adjacent signed cosine and analytic per-step least-squares optimal coefficient as diagnostics, not as a deployment result.
- Invalid cheap proxy to avoid: Interpreting saved `e_{t-1}` as available in a real quantized inference run; correcting latent states rather than model outputs; summing unpaired trajectories; claiming improved generated video from a one-step teacher-forced metric.
- Expected outcomes and decisions: If both fixed coefficients reduce error consistently, temporal signed coherence is a promising target for calibration/correction research. If gains are inconsistent, inspect error-norm ratios and timestep dependence before proposing a fixed-coefficient runtime method. If neither helps, the observed cosine may not translate into useful next-step error prediction.
- Stop condition: Reproducible per-step and pooled finite results for both prompts and modalities, or a documented raw-file integrity/alignment failure. This is a read-only oracle calculation; no new video generation is required.
- Evidence classification: Supporting mechanism evidence for E105/E106, not an inference-speed or deployment-quality result.
