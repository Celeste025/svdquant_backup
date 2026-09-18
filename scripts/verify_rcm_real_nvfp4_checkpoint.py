#!/usr/bin/env python3
"""Validate an rCM real-NVFP4 SVDQuant checkpoint and write its manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(32, 64), required=True)
    parser.add_argument("--smooth-grids", type=int, choices=(10, 20), required=True)
    parser.add_argument("--lowrank-iters", type=int, default=100)
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ("model.pt", "scale.pt", "wgts.pt"):
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(args.checkpoint / name)
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)

    sys.path.insert(0, str(ROOT / "scripts"))
    from infer_rcm_wan_4step import load_quantized_transformer

    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.transformer = WanTransformer3DModel.from_pretrained(args.transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    recipe = f"rcm-wan-real-nvfp4-r{args.rank}-g{args.smooth_grids}"
    load_quantized_transformer(pipe, args.checkpoint, args.model, recipe)
    linears = [(name, m) for name, m in pipe.transformer.named_modules() if isinstance(m, torch.nn.Linear)]
    targets = [(name, m) for name, m in linears if name.startswith("blocks.")]
    # rCM has 300 quantized block projections plus six native condition/embedder
    # Linears.  Low-rank compensation is installed through hooks and is not a
    # registered transformer child module, so count target projections directly.
    if len(targets) != 300 or len(linears) != 306:
        raise RuntimeError(f"expected 300 block targets and 306 total Linears, got {len(targets)} / {len(linears)}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)

    def to_cuda(value):
        if torch.is_tensor(value): return value.to("cuda")
        if isinstance(value, list): return [to_cuda(v) for v in value]
        if isinstance(value, tuple): return tuple(to_cuda(v) for v in value)
        if isinstance(value, dict): return {k: to_cuda(v) for k, v in value.items()}
        return value

    with torch.inference_mode():
        output = pipe.transformer(*to_cuda(cache["input_args"]), **to_cuda(cache["input_kwargs"]))[0]
    if not torch.isfinite(output).all():
        raise RuntimeError("non-finite cached forward")
    manifest = {
        "format": "rcm-wan-real-nvfp4-svdquant-v1",
        "state": "complete",
        "quantization": {"weights": "real NVFP4 E2M1, group16", "activations": "dynamic real NVFP4 E2M1, group16"},
        "svdquant": {"rank": args.rank, "smooth_grids": args.smooth_grids, "lowrank_iters": args.lowrank_iters},
        "calibration": {"cache_root": str(args.cache.parent.parent), "samples": args.calib_samples, "rcm_steps": 4, "frames": 77, "height": 480, "width": 832, "sigma_max": 80.0},
        "smoke": args.smoke,
        "model": str(args.model), "transformer": str(args.transformer),
        "validation": {"quantization_target_linears": len(targets), "total_native_linears": len(linears), "cached_forward_finite": True},
    }
    (args.checkpoint / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
