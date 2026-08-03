#!/usr/bin/env python3
"""Collect sampled Wan Linear inputs for output-error SVDQuant calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from compare_wan_svdquant_fake import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    REMAIN_FP,
    make_pipeline,
    seed_everything,
)


class InputSampler:
    def __init__(self, tokens_per_call: int) -> None:
        self.tokens_per_call = tokens_per_call
        self.samples: list[torch.Tensor] = []
        self.calls = 0

    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        count = min(self.tokens_per_call, x.shape[0])
        # Stratified positions cover the complete spatial-temporal/text sequence
        # deterministically without perturbing any RNG used by generation.
        positions = torch.linspace(0, x.shape[0] - 1, count, device=x.device).long()
        self.samples.append(x.index_select(0, positions).to(device="cpu", dtype=torch.bfloat16))
        self.calls += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wan_svdquant_fake/calib_seed44_tokens.pt")
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--tokens-per-call", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = make_pipeline(args.model)
    samplers: dict[str, InputSampler] = {}
    handles = []
    for name, module in pipe.transformer.named_modules():
        if (
            isinstance(module, nn.Linear)
            and not any(pattern in name for pattern in REMAIN_FP)
            and name.startswith("blocks.")
        ):
            sampler = InputSampler(args.tokens_per_call)
            samplers[name] = sampler
            handles.append(module.register_forward_pre_hook(sampler))
    if len(samplers) != 300:
        raise RuntimeError(f"expected 300 Linear modules, found {len(samplers)}")

    pipe.enable_model_cpu_offload()
    seed_everything(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        output_type="latent",
    )
    for handle in handles:
        handle.remove()

    calibration = {name: torch.cat(sampler.samples, dim=0) for name, sampler in samplers.items()}
    calls = {name: sampler.calls for name, sampler in samplers.items()}
    expected_rows = args.steps * 2 * args.tokens_per_call
    bad = {
        name: list(tensor.shape)
        for name, tensor in calibration.items()
        if tensor.shape[0] != expected_rows
    }
    if bad:
        raise RuntimeError(f"unexpected calibration shapes: {bad}")

    payload = {
        "format": "wan-svdquant-token-inputs-v1",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "flow_shift": 3.0,
        "tokens_per_call": args.tokens_per_call,
        "calls": calls,
        "inputs": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        key: value
        for key, value in payload.items()
        if key not in ("inputs", "calls")
    }
    summary["num_layers"] = len(calibration)
    summary["rows_per_layer"] = expected_rows
    summary["bytes"] = args.output.stat().st_size
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
