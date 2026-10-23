# Isolated SageAttention3 H3 noncausal padding-mask fix

Source forked from the E098 SM110 benchmark copy, itself derived from the
official SageAttention3 Blackwell checkout at commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. The upstream copy and
the E098 binary remain untouched. This fork maps each physical QK score
column through K's within-32 permutation before applying the logical
unpadded-length mask. Only the noncausal mask is changed; causal behavior
has not been validated. Build with `scripts/research/build_e116_sage3_maskfix.sh`.

The `csrc/cutlass` symlink points to the existing local CUTLASS source; no
new network dependency is needed.
