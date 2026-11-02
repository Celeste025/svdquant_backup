# E121 H3 split mixed attention generation

Use the BF16 MiniMax-H3 model and E113 race prompt/seed (prompt ID 16, seed
20026), 576×1024, 124 frames, 20 steps, CFG 1, tiled decode. Compare the
same full generation with (1) original BF16 attention, (2) corrected Sage3
FP4, and (3) E120 split P8×V8 mixed attention at threshold 0.10. Replace only
the 50 main block attention helpers; retain BF16 linears, refiner attention,
and padding-segment attention. Verify one cached DiT call before generation.

Report denoise wall time (including model loading), sum of 20 DiT times,
decode wall time, and total denoise-plus-decode wall time separately. The
three runs are sequential on one otherwise-idle RTX PRO 5000 to avoid shared
host I/O contention. Preserve per-arm logs and media; do not reuse E113
timings as a paired control. Only claim end-to-end speedup from measured full
generation, not from isolated attention microbenchmarks.
