#!/usr/bin/env python3
"""Run the official Nunchaku FLUX.1-dev W4A4 checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel


BASE_MODEL = "black-forest-labs/FLUX.1-dev"
QUANT_MODEL = "mit-han-lab/svdq-int4-flux.1-dev"
PROMPT = "A cat holding a sign that says hello world"
SEEDS = (42, 43, 44)


def main() -> None:
    output_dir = Path("outputs/flux_dev_official_w4a4")
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        QUANT_MODEL, device="cuda:0"
    )
    pipeline = FluxPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")

    torch.cuda.reset_peak_memory_stats()
    records = []
    for seed in SEEDS:
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            image = pipeline(
                PROMPT,
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=3.5,
                generator=generator,
            ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        path = output_dir / f"official_w4a4_seed{seed}.png"
        image.save(path)
        records.append({"seed": seed, "seconds": elapsed, "path": str(path)})
        print(f"seed={seed} seconds={elapsed:.3f} path={path}", flush=True)

    report = {
        "base_model": BASE_MODEL,
        "quantized_transformer": QUANT_MODEL,
        "prompt": PROMPT,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
        "weight_bits": 4,
        "activation_bits": 4,
        "svd_rank": 32,
        "records": records,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    (output_dir / "official_w4a4_metadata.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
