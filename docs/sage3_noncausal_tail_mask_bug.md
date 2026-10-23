# SageAttention3 noncausal K-tail mask: local finding and repair

The Sage3 Blackwell build audited in E115 packs K with a within-32-row
permutation. In its noncausal attention path, the original score mask
compared an MMA **physical column** directly with the **logical** unpadded
K length. These coordinate systems disagree at a non-32-aligned tail.

For a 54-token input padded to 128, four valid keys were masked and four
padding positions were admitted. The measured old Sage3 output versus a
logical-mask Triton reference had NMSE `0.00646812` (reference-output
energy denominator). A 32-token aligned control was identical. On the
9,482-token H3 main segment, the old pure output versus that reference had
NMSE `1.73854e-5`. E115 confirmed the same behavior with an isolated
unmodified upstream-derived Sage3 build; this is not inferred only from
our modified kernel.

The E116 fork repairs the first noncausal score mask by mapping the
physical K column back to its logical token index before the length
comparison. The corrected code lives in both
`third_party/sage3_h3_maskfix/sageattn3/blackwell/mainloop_tma_ws.h`
and `third_party/sage3_native_mixed/sageattn3/blackwell/mainloop_tma_ws.h`.
It does not alter causal masking. The old Sage3 source/binary was not
overwritten, and the separate corrected modules are used in E120.

After repair, the 54-token pure output was byte-identical to the
logical-mask reference; the 9,482-token pure and native mixed outputs
were within NMSE `1.13e-8` and `1.14e-8` of their matching references.
The repaired pure online path was about 2.07% slower than the old one in
the three-process same-GPU A/B. The results are for captured H3 Q/K/V,
noncausal attention and this local Sage3 revision; they do not establish
the status of any later upstream release.

Evidence and reproduction details:

- `results/research/E115/README.md`: discovery, K permutation and old
  mask comparison; `official_unmodified_mask_gpu1.json` is the direct
  old-build control.
- `results/research/E116/README.md`: corrected pure/mixed kernels,
  32/54/9482-token A/B, BF16-reference error and speed.
- `scripts/research/verify_e115_official_sage3_mask.py` and
  `scripts/research/e116_sage3_maskfix_ab.py`: numerical checks. They
  require the unversioned H3 QKV capture at the path recorded in the
  scripts. The latter also requires the dense original V router and
  the E113 Triton reference; E120's generic call wrappers do not.
- `scripts/research/diagnose_e115_sage3_fp4_alignment.py` is the broader
  quantizer/alignment audit. It additionally imports the locally checked-out
  `third_party/MixAttention` reference, which is not included in this branch.
