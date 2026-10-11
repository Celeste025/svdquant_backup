# E087 timestep-aware smoothing probe

Exploratory configuration: 2 fitting prompts and 2 disjoint validation prompts; 20 timesteps; 2 selected linears; smoothing grid=10. Objective: parent DiT block endpoint NMSE for a single W4A4+smoothing candidate.

Scope note: One-hour exploratory probe: two attention linears share block-13 replay; two fit and two independent validation prompts. Results are directional due to n=2 validation prompts.

|layer|distinct best α|local vs global gain|mean |log(s_local/s_global)||min cosine|
|---|---|---:|---:|---:|
|blocks.13.attn.qkv_proj|[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]|-0.777%|0.098|0.95615|
|blocks.13.attn.out_proj|[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]|10.816%|1.025|0.84978|
