#!/usr/bin/env python3
"""Benchmark resident-GPU FLUX.1-schnell DiT latency without T5 or VAE."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline

from compare_flux import BASE_MODEL, INT4_MODEL


def process_gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        pid = os.getpid()
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and int(fields[0]) == pid:
                return int(fields[1])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def load_pipeline(precision: str, device: str) -> FluxPipeline:
    common = {
        "text_encoder": None,
        "text_encoder_2": None,
        "tokenizer": None,
        "tokenizer_2": None,
        "vae": None,
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if precision == "w4a4":
        from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(INT4_MODEL, device=device)
        pipe = FluxPipeline.from_pretrained(BASE_MODEL, transformer=transformer, **common)
    else:
        pipe = FluxPipeline.from_pretrained(BASE_MODEL, **common)
    return pipe.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("precision", choices=("w4a4", "bf16"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flux_comparison"))
    args = parser.parse_args()

    embedding_path = args.output_dir / "bf16_prompt_embeddings.pt"
    payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
    pipe = load_pipeline(args.precision, args.device)
    prompt_embeds = payload["prompt_embeds"].to(args.device)
    pooled_prompt_embeds = payload["pooled_prompt_embeds"].to(args.device)
    pipe.set_progress_bar_config(disable=True)

    def invoke(seed: int) -> None:
        generator = torch.Generator(device=args.device).manual_seed(seed)
        pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=1024,
            width=1024,
            generator=generator,
            output_type="latent",
        )
        torch.cuda.synchronize()

    for index in range(args.warmup):
        invoke(1000 + index)

    resident_mib = process_gpu_memory_mib()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for index in range(args.runs):
        start = time.perf_counter()
        invoke(2000 + index)
        timings.append(time.perf_counter() - start)

    result = {
        "precision": args.precision,
        "warmup": args.warmup,
        "runs": args.runs,
        "height": 1024,
        "width": 1024,
        "steps": 4,
        "timings_seconds": [round(value, 6) for value in timings],
        "mean_seconds": round(sum(timings) / len(timings), 6),
        "mean_step_ms": round(sum(timings) / len(timings) / 4 * 1000, 3),
        "resident_process_gpu_mib": resident_mib,
        "torch_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    path = args.output_dir / f"benchmark_{args.precision}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
