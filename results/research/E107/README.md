# E107 — previous-step oracle output-error subtraction

Read-only diagnostic using all 20 hash-verified saved paired complete-DiT calls from E103 race and E104 clap0. For each modality and step `t=1..19`, define `e_t = SVDQuant_t - BF16_t` on the same saved BF16-trajectory input and evaluate `e_t - k e_(t-1)` for fixed `k=0.5,0.8`. NMSE uses the current step's BF16 output squared energy. The pooled values below sum squared error numerators and BF16 energy denominators over steps 1–19; step 0 has no previous error and is excluded.

| Case | Modality | Baseline pooled NMSE | `k=0.5` pooled NMSE / relative drop | `k=0.8` pooled NMSE / relative drop | Steps improved, k=0.5 / 0.8 |
|---|---|---:|---:|---:|---:|
| Race | Video | 8.398% | 5.399% / 35.7% | 5.731% / 31.8% | 19/19, 19/19 |
| Race | Audio | 4.192% | 1.941% / 53.7% | 1.584% / 62.2% | 19/19, 19/19 |
| Clap | Video | 2.334% | 1.644% / 29.6% | 1.797% / 23.0% | 18/19, 18/19 |
| Clap | Audio | 1.734% | 1.148% / 33.8% | 1.207% / 30.4% | 18/19, 16/19 |

The per-step least-squares optimal coefficient `dot(e_t,e_(t-1))/||e_(t-1)||²` has median 0.592 (race video), 0.726 (race audio), 0.582 (clap video), and 0.633 (clap audio). This explains why 0.8 is not universally better than 0.5. The fixed coefficients worsen clap video step 3; fixed 0.8 also worsens clap audio steps 2, 5, and 19. Across steps, a positive signed cosine alone does not guarantee a particular `k` helps: the preceding error's norm also matters.

An exact quadratic sweep over pooled steps 1–19 evaluated `k=0..1.5` in increments of 0.01. The pooled oracle minima occur at `k=0.588` (race video, NMSE 5.331%), `0.794` (race audio, 1.584%), `0.542` (clap video, 1.640%), and `0.592` (clap audio, 1.133%). These are *pooled in-sample* minima, distinct from the median per-step optimum above; they must not be presented as a tuned deployment coefficient. The k=0, 0.5, and 0.8 sweep values were checked against the independently stored summary values to numerical precision. See the [NMSE-versus-k figure](figures/k_sweep_pooled_nmse.png), [full sweep CSV](k_sweep.csv), and [sweep JSON](k_sweep.json).

This is **not** an implementable correction as measured: it uses the true previous BF16-vs-quantized error, which an ordinary quantized inference run does not know. It also does not rerun the scheduler after correction or establish an improved generated video. Its valid conclusion is narrower: on these two saved BF16 trajectories, substantial next-step complete-DiT error is linearly predictable from the preceding true error. Any practical method needs a deployable error estimator or calibration target and a closed-loop held-out validation.

Raw per-step results: [CSV](per_step.csv) and [JSON](previous_error_subtraction.json). Source: `scripts/research/eval_e107_h3_previous_error_subtraction.py`; source manifest and raw-file SHA256 are checked by the script.
