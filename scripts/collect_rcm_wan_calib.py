#!/usr/bin/env python3
"""Collect DeepCompressor-compatible caches along the rCM Wan 1--4 step path."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from diffusers import WanPipeline
from deepcompressor.app.diffusion.dataset.collect.utils import CollectHook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=77)
    args = parser.parse_args()

    cache_dir = args.output / "caches"
    if cache_dir.exists() and any(cache_dir.glob("*.pt")):
        raise FileExistsError(f"Refusing to overwrite existing caches: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_map = yaml.safe_load(args.prompts.read_text())
    prompts = list(prompt_map.items())[: args.num_prompts]
    if len(prompts) != args.num_prompts:
        raise ValueError(f"Requested {args.num_prompts} prompts, found {len(prompts)}")

    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    embeddings = []
    with torch.inference_mode():
        for _, prompt in prompts:
            embedding, _ = pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                max_sequence_length=512,
                device=device,
                dtype=pipe.transformer.dtype,
            )
            embeddings.append(embedding.cpu())
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    hook = CollectHook(caches=[])
    handle = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
    mid_t = [1.5, 1.4, 1.0][: args.steps - 1]
    trig_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0.0], dtype=torch.float64, device=device)
    t_steps = torch.sin(trig_steps) / (torch.cos(trig_steps) + torch.sin(trig_steps))
    try:
        with torch.inference_mode():
            for prompt_index, (filename, _) in enumerate(prompts):
                generator = torch.Generator(device=device).manual_seed(args.seed + prompt_index)
                latents = pipe.prepare_latents(
                    batch_size=1,
                    num_channels_latents=pipe.transformer.config.in_channels,
                    height=args.height,
                    width=args.width,
                    num_frames=args.frames,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                ).to(torch.float64) * t_steps[0]
                embedding = embeddings[prompt_index].to(device=device, dtype=pipe.transformer.dtype)
                for step, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                    timestep = torch.full(
                        (1,), t_cur.float().item() * 1000, device=device, dtype=pipe.transformer.dtype
                    )
                    velocity = pipe.transformer(
                        hidden_states=latents.to(pipe.transformer.dtype),
                        timestep=timestep,
                        encoder_hidden_states=embedding,
                        return_dict=False,
                    )[0].to(torch.float64)
                    if len(hook.caches) != 1:
                        raise RuntimeError(f"Expected one captured forward, got {len(hook.caches)}")
                    cache = hook.caches.pop()
                    cache.update(filename=str(filename), step=step, guidance=0)
                    torch.save(cache, cache_dir / f"{filename}-{step:05d}-0.pt")
                    latents = (1 - t_next) * (latents - t_cur * velocity) + t_next * torch.randn(
                        latents.shape, dtype=torch.float32, device=device, generator=generator
                    )
                    del velocity
                print(f"collected {prompt_index + 1}/{len(prompts)}: {filename}", flush=True)
    finally:
        handle.remove()
    count = len(list(cache_dir.glob("*.pt")))
    expected = args.num_prompts * args.steps
    if count != expected:
        raise RuntimeError(f"Cache count mismatch: {count} != {expected}")
    print(f"Saved {count} caches to {cache_dir}")


if __name__ == "__main__":
    main()
