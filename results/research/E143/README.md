# E143 — E142 clap/block49 non-monotone NMSE audit

Target: `clap_s10_b49`, full H3 Q/K/V capture, video-query output only.
The diagnostic script is `scripts/research/diagnose_e142_clap_anomaly.py`; raw
results are in `clap_s10_b49_diagnostic.json`.

## Conclusion

The decreasing mixed-FP4 NMSE with increasing sparsity is reproducible and is
not caused by NaN/Inf, output overflow, nondeterminism, a changing denominator,
or a plotting/statistics error.  At fixed DraftMap support, BF16 sparsification
error increases monotonically, while the conditional FP4 arithmetic error falls
more quickly because fewer low-score interactions are evaluated in FP4.

| skip | BF16-support NMSE | conditional quantization NMSE | cross term | total mixed NMSE |
|---:|---:|---:|---:|---:|
| 0% | 0.0173% | 10.8460% | -0.0204% | 10.8429% |
| 30% | 1.0956% | 7.6348% | -0.0659% | 8.6644% |
| 50% | 1.7951% | 5.5581% | -0.0991% | 7.2541% |
| 70% | 2.5971% | 3.6535% | -0.1355% | 6.1150% |
| 80% | 3.1853% | 2.7082% | -0.1676% | 5.7258% |

The cross term is too small to explain the reversal.  The dominant cause is the
large reduction in conditional FP4 error.

The case is unusually FP4-sensitive and head-localized.  At 0% skip with 75%
FP4 / 25% INT8 retained arithmetic, median per-head NMSE is only 0.262%, but
heads 47 and 11 reach 60.95% and 54.90%.  These are also among the six heads
with exceptional K magnitudes (K RMS about 28–33 and maxima 278–378; most other
heads have no values above half of the official NVFP4 global range).

The official layer-49 tensor scales are Q=0.0782645, K=0.1598772, V=0.4821429.
All observed values fit the nominal `2688 * scale` range, so there is no normal
full-range saturation at the default scale.  A component-wise scale sweep shows
that halving only K scale (thus clipping the extreme K tail) reduces dense mixed
NMSE from 10.8429% to 3.9355%; changing V scale has essentially no effect, and
changing Q scale only modestly affects it.  This identifies extreme K/logit
sensitivity—not V/PV arithmetic—as the special numerical feature.

All audited outputs were finite.  Output maxima remained 458–464 versus BF16
460.  Repeated default runs at 0% and 80% skip were bitwise identical.  Pure
INT8 at 0% skip was 0.1793% NMSE, further excluding a general routing/indexing or
softmax failure.  The result therefore supports a precision-allocation / static
K-scale failure mode rather than a kernel correctness bug.

## Follow-up checkpoint

Do not generalize from this one prompt/layer capture yet.  Later experiments
should determine (1) how frequently the K/logit FP4 failure occurs across
prompts, timesteps and layers, and (2) whether the same anomalous heads
(especially 47, 11, 9, 25 and 30 here) recur across prompts.  A useful next
comparison is per-head K-scale or head-selective INT8 versus the current
tile-score precision allocation.
