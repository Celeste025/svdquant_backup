#!/usr/bin/env python3
"""Compare SVDQuant W4A4 and BF16 FLUX.1 with identical text embeddings."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline


# Byte-for-byte Diffusers mirror of the gated upstream repository. The server's
# cached Hugging Face credential is expired, while this Apache-2.0 mirror is
# readable without authentication.
BASE_MODEL = "frankjoshua/FLUX.1-schnell"
INT4_MODEL = "mit-han-lab/svdq-int4-flux.1-schnell"
DEFAULT_PROMPT = (
    "A cinematic photograph of a red fox sitting beside a small weathered wooden sign "
    "that clearly reads 'SVDQuant', in a misty pine forest at sunrise, soft volumetric "
    "light, realistic fur, shallow depth of field, highly detailed"
)
DEFAULT_SEEDS = (42, 43, 44)


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def memory_gib() -> dict[str, float]:
    return {
        "allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def encode_prompt(prompt: str, embedding_path: Path, device: str) -> None:
    """Run the original BF16 CLIP/T5 encoders, then save their outputs on CPU."""
    print(f"Loading BF16 text encoders from {BASE_MODEL}")
    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
    )
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=torch.device(device),
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    payload = {
        "prompt": prompt,
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
        "text_ids": text_ids.cpu(),
        "text_encoder_peak": memory_gib(),
    }
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, embedding_path)
    print(f"Saved shared BF16 prompt embeddings to {embedding_path}")
    print(f"Text encoder CUDA peak: {payload['text_encoder_peak']}")

    del prompt_embeds, pooled_prompt_embeds, text_ids, pipe
    clear_cuda()


def load_generation_pipeline(precision: str, device: str) -> FluxPipeline:
    if precision == "w4a4":
        from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(INT4_MODEL, device=device)
        pipe = FluxPipeline.from_pretrained(
            BASE_MODEL,
            transformer=transformer,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            torch_dtype=torch.bfloat16,
        )
        pipe.to(device)
    else:
        pipe = FluxPipeline.from_pretrained(
            BASE_MODEL,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            torch_dtype=torch.bfloat16,
        )
        # A BF16 FLUX DiT nearly fills a 24 GiB card. Sequential model CPU
        # offload keeps the transformer and VAE from residing on CUDA together.
        pipe.enable_model_cpu_offload(gpu_id=torch.device(device).index or 0)

    pipe.set_progress_bar_config(disable=False)
    return pipe


def generate(
    precision: str,
    embedding_path: Path,
    output_dir: Path,
    seeds: list[int],
    device: str,
) -> None:
    payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {precision} generation pipeline")
    pipe = load_generation_pipeline(precision, device)
    torch.cuda.reset_peak_memory_stats()

    records = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        start = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt_embeds=payload["prompt_embeds"].to(device),
                pooled_prompt_embeds=payload["pooled_prompt_embeds"].to(device),
                num_inference_steps=4,
                guidance_scale=0.0,
                height=1024,
                width=1024,
                generator=generator,
            ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        image_path = output_dir / f"{precision}_seed{seed}.png"
        image.save(image_path)
        records.append({"seed": seed, "seconds": round(elapsed, 3), "path": str(image_path)})
        print(f"Saved {image_path} ({elapsed:.2f}s)")

    metadata = {
        "base_model": BASE_MODEL,
        "quantized_model": INT4_MODEL if precision == "w4a4" else None,
        "precision": precision,
        "prompt": payload["prompt"],
        "seeds": seeds,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "shared_embedding_file": str(embedding_path),
        "text_encoder_peak": payload["text_encoder_peak"],
        "generation_peak": memory_gib(),
        "images": records,
    }
    metadata_path = output_dir / f"{precision}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Generation CUDA peak: {metadata['generation_peak']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("encode", "w4a4", "bf16"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flux_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_path = args.output_dir / "bf16_prompt_embeddings.pt"
    if args.stage == "encode":
        encode_prompt(args.prompt, embedding_path, args.device)
    else:
        if not embedding_path.exists():
            raise FileNotFoundError(f"Run the encode stage first: missing {embedding_path}")
        generate(args.stage, embedding_path, args.output_dir, args.seeds, args.device)


if __name__ == "__main__":
    main()
