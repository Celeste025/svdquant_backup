# E103 — H3 paired DiT-output error-pattern capture

- Experiment ID: E103.
- Mode: exploration.
- Problem addressed: Does rank-32 SVDQuant create structured complete-DiT output error, rather than spatially/channel-uniform noise, on a real H3 video-generation trajectory?
- Hypothesis: Even at comparable aggregate NMSE, error may concentrate in output channels, latent video frames or spatial regions, and exhibit adjacent-token correlation.
- Decision informed: Which error-structure variables deserve a controlled equal-NMSE generation-quality experiment next.
- Minimum valid setup: Reuse the existing BF16 race prompt 16 / seed 20026 / 576×1024 / 124-frame / 20-step cached DiT calls and standard outputs. Replay the same calls through the existing rank-32 SVDQuant dynamic-scale runtime. Save every step's BF16 and quantized video/audio outputs plus FP32 signed differences. Verify prompt, step count, shapes, finite values, per-step NMSE against E101, and video-token order before spatial interpretation.
- Invalid cheap proxy to avoid: Comparing separate free rollouts, studying only a cropped tensor, averaging away the channel/spatial structure, or calling one trajectory a general effect.
- Expected outcomes: Structured channel/space/time concentration; mostly unstructured errors; or mixed structure that varies by denoising step.
- Decision for each outcome: If structured, construct controlled equal-NMSE perturbations retaining or scrambling the identified structure in a later experiment; if unstructured, prioritize magnitude/timestep over location; if mixed, isolate the dominant axis.
- Stop condition: Complete validated 20-step raw capture, initial channel/spatiotemporal/distribution summaries, and inspectable figures. No claim yet about perceptual generation quality or causal impact of patterns at equal NMSE.
- Evidence: Necessary for choosing the follow-up intervention; insufficient by itself to establish quality differences.
