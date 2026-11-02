# E188 real trajectory replay checkpoint

User steering: existing correctness validation is sufficient; do not make it
overly complex/strict. E189 full-generation correctness script was prepared
locally but NEVER uploaded/launched. Do not run it as a completion gate.
Future optimization should prioritize performance with lightweight finite /
numerical checks. Preserve90%NVFP4+10%INT8 scope, plus denseINT8 first capture.

E187 completed6real480p124f QKV captures:layers0/25/49 atsteps0/10,15795tokens.
Manifest:results/research/E187/manifest.json; no further generation task running.
E188 replay uses same-layer step0route capture and actualstep10reuse input.
GPU1,10warmups30timedcalls perpath,forward/reversepathorders,originalA/current/
originalB. Outputs and routehashes allbitwise equal for all3layers and paths.

|Layer|Capture original/current ms|Capture reduction|Reuse original/current ms|Reuse reduction|
|---:|---:|---:|---:|---:|
|0|38.1133 /33.2119|12.86%|29.7535 /28.9819|2.59%|
|25|38.1918 /33.2479|12.94%|29.7636 /29.0140|2.52%|
|49|37.8229 /32.9632|12.85%|29.5984 /28.8844|2.41%|

Original means averageA/B. Unchangedproxy timings changed+0.2–0.4% in current
round,so timingcontrolreasonable. Source and binary unchanged fromE186;
current SHA2561f52b6dffc744d3bc175beb53a90d49124e66ee264814aaf39fab766e92949a1.
ABA EXITtrap restored current and no tmux task remained at completion.

The full goal is not marked complete: capture gain nowwellsupported, reuse
modest. Next profiling should focus on mainkernel hot source/PC counters,
not further expensive end-to-end correctness checks.
