# E132 — race token-error patterns after 10% BF16 recomputation

Both arms replayed the same 20 cached BF16 teacher calls for race/p016. The
native NVFP4 W4A4 implementation and original BF16 weights are those of E128;
the recompute arm uses the E131 per-linear, per-step random 10% token policy.
Attention stays BF16. We compare complete DiT video output against each call's
BF16 output, then sum squared errors over 96 output channels to form a
37×18×32 token-energy map for each step. This is teacher-forced output error,
not the difference between free-rollout latents or decoded pixels.

Across 19 adjacent step pairs:

| Metric | W4A4 | W4A4 + 10% recompute |
|---|---:|---:|
| Pearson correlation of log token error energy | 0.7339 | 0.7134 |
| Cosine similarity of signed error tensors | 0.6353 | 0.6081 |
| Jaccard of top 1% video-error tokens | 0.1724 | 0.1667 |

The random-position expected Jaccard for two independent 1% sets is about
0.005. Thus temporal structure remains, though it weakens slightly. At the
same step, cross-arm log token-energy correlation averages 0.8824 (range
0.8372–0.9204). On average 54.4% of W4A4's top-1% tokens remain in the
recompute arm's top-1% set; the other 45.6% change rank membership. Both arms'
top 1% carry only about 5.5% of total video-output error energy, so this
threshold is a tail-location diagnostic, not a majority-energy description.

The recompute arm reduces error energy by a mean of 13.0% per step (step 0:
20.1%; step 19: 10.4%). At the original W4A4 top-1% positions, mean reduction
is 24.5%. Results are one prompt and one random-mask seed. See
`comparison.json` for every step and the two `*_token_maps.npz` files for the
full token-energy grids.
