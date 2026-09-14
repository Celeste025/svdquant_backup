#!/usr/bin/env python3
"""Profile the Wan VAE decoder memory at the rCM 480p latent resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import WanPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []

    def record(stage: str, **extra: object) -> None:
        torch.cuda.synchronize()
        row = {
            "stage": stage,
            "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
            "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            **extra,
        }
        rows.append(row)
        print(row, flush=True)

    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    vae = pipe.vae.to("cuda").eval()
    del pipe
    record("vae_only_loaded")
    latents = torch.randn((1, 16, 5, 60, 104), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        projected = vae.post_quant_conv(latents)
        record("post_quant_conv", shape=list(projected.shape))
        vae.clear_cache()
        for index in range(projected.shape[2]):
            vae._conv_idx = [0]
            chunk = vae.decoder(projected[:, :, index : index + 1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx)
            record("decoder_chunk", index=index, shape=list(chunk.shape))
            del chunk
            torch.cuda.empty_cache()
        vae.clear_cache()
        record("after_cache_clear")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
