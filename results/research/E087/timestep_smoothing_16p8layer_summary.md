# E087 timestep-aware smoothing probe

Exploratory configuration: 8 fitting prompts and 8 disjoint validation prompts; 20 timesteps; 8 selected linears; smoothing grid=10. Objective: parent DiT block endpoint NMSE for a single W4A4+smoothing candidate.

Scope note: Main probe: 16 prompts x 20 timesteps, eight linears from four depths and all four target projection types. Eight prompts fit smoothing; eight disjoint prompts measure held-out parent-block-end NMSE.

|layer|distinct best α|local vs global gain|mean |log(s_local/s_global)||min cosine|
|---|---|---:|---:|---:|
|blocks.3.attn.qkv_proj|[0.4, 0.5, 0.6, 0.7]|-0.442%|0.084|0.98324|
|blocks.3.attn.out_proj|[0.3, 0.4, 0.5, 0.6, 0.7]|0.041%|0.149|0.97614|
|blocks.13.attn.qkv_proj|[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]|-18.496%|0.152|0.93937|
|blocks.13.attn.out_proj|[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]|5.187%|0.389|0.91325|
|blocks.25.mlp.fc1|[0.3, 0.4, 0.5]|1.301%|0.066|0.99475|
|blocks.25.mlp.fc2|[0.2, 0.3, 0.4]|0.801%|0.468|0.98601|
|blocks.46.mlp.fc1|[0.5, 0.6, 0.7, 0.8]|0.454%|0.187|0.98938|
|blocks.46.mlp.fc2|[0.3, 0.4, 0.5]|3.086%|0.503|0.96921|
