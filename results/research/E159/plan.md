Four cases: p025, p048 (5s); p044, p007 (10s). No 15s cases.
Two arms each, BF16 linears, Q128, skip0 / NVFP4 .9 / INT8 .1.
Reuse existing E150 BF16 latents; verify initial video/audio noise hashes.
Both arms use step0 dense INT8 and layer0 BF16 throughout. Reuse additionally captures step0 map. Steps1..19 use 90% NVFP4 + 10% INT8: proxy rebuilds map; reuse loads captured route.
Final video latent NMSE to BF16 is primary metric. First-step output policy is aligned; later routing policies intentionally differ.
