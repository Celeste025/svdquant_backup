# E148: p025, 75% sparse / 15% NVFP4 / 10% INT8

Reuse E124/E146/E147 prompt, seed, initial noise and 576x1024, 124-frame,
20-step setup. BF16 linears; first two timesteps dense, block 0 dense on
remaining timesteps; standard Anemoi prefix INT8 and per-layer NVFP4 scales.
Compare video and metrics with BF16, E146 80/0/20 and E147 74.3/19.3/6.4.
