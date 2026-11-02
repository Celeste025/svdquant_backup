# E147 — paper allocations on H3 p025

Compare 80/0/20 to 74.3/19.3/6.4, proportions in skip/NVFP4/INT8
of video-block pairs. Reuse E146 80/0/20 and E124 BF16; same prompt 25,
seed 2141, 576x1024, 124 frames, 20 steps, BF16 linears, first two steps
dense and block 0 dense thereafter. Prefix attention uses INT8. Measure
DiT time, paired RGB NMSE/LPIPS, and operator latency on saved real QKV.
