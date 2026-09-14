#!/usr/bin/env python3
"""Convert the public rCM Wan-2.1 T2V DiT checkpoint to Diffusers naming.

The rCM and Diffusers 1.3B backbones contain the same 825 tensors.  This
script intentionally converts only the DiT: tokenizer, UMT5 and VAE are
reused from an already validated local Wan2.1 Diffusers pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def target_key(source_key: str) -> str:
    root = {
        "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
        "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
        "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
        "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
        "time_projection.1.": "condition_embedder.time_proj.",
        "head.modulation": "scale_shift_table",
        "head.head.": "proj_out.",
    }
    for old, new in root.items():
        if source_key.startswith(old):
            return new + source_key[len(old) :]

    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", source_key)
    if not match:
        return source_key
    index, suffix = match.groups()
    replacements = {
        "modulation": "scale_shift_table",
        "self_attn.norm_q.": "attn1.norm_q.",
        "self_attn.norm_k.": "attn1.norm_k.",
        "self_attn.q.": "attn1.to_q.",
        "self_attn.k.": "attn1.to_k.",
        "self_attn.v.": "attn1.to_v.",
        "self_attn.o.": "attn1.to_out.0.",
        "cross_attn.norm_q.": "attn2.norm_q.",
        "cross_attn.norm_k.": "attn2.norm_k.",
        "cross_attn.q.": "attn2.to_q.",
        "cross_attn.k.": "attn2.to_k.",
        "cross_attn.v.": "attn2.to_v.",
        "cross_attn.o.": "attn2.to_out.0.",
        "norm3.": "norm2.",
        "ffn.0.": "ffn.net.0.proj.",
        "ffn.2.": "ffn.net.2.",
    }
    for old, new in replacements.items():
        if suffix.startswith(old):
            return f"blocks.{index}.{new}{suffix[len(old):]}"
    raise KeyError(f"No Diffusers mapping for rCM key: {source_key}")


def expected_shapes(transformer_dir: Path) -> dict[str, tuple[int, ...]]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    files = (
        sorted(transformer_dir.glob("*.safetensors"))
        if not index_path.exists()
        else sorted({transformer_dir / name for name in json.loads(index_path.read_text())["weight_map"].values()})
    )
    result: dict[str, tuple[int, ...]] = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                result[key] = tuple(handle.get_slice(key).get_shape())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcm-ckpt", type=Path, required=True)
    parser.add_argument("--base-pipeline", type=Path, required=True)
    parser.add_argument("--output-pipeline", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.rcm_ckpt, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state = {key.removeprefix("net."): value for key, value in state.items()}
    expected = expected_shapes(args.base_pipeline / "transformer")
    converted: dict[str, torch.Tensor] = {}
    for source, tensor in state.items():
        # Training counters are bundled with the public .pt but are not model
        # parameters and have no Diffusers counterpart.
        if source.startswith("accum_"):
            continue
        destination = target_key(source)
        if destination == "patch_embedding.weight":
            tensor = tensor.reshape(1536, 16, 1, 2, 2)
        if destination not in expected:
            raise KeyError(f"Unexpected converted key: {source} -> {destination}")
        if tuple(tensor.shape) != expected[destination]:
            raise ValueError(f"Shape mismatch {source} -> {destination}: {tuple(tensor.shape)} != {expected[destination]}")
        if destination in converted:
            raise KeyError(f"Duplicate destination key: {destination}")
        converted[destination] = tensor.contiguous()
    missing = sorted(set(expected) - set(converted))
    if missing or len(converted) != len(expected):
        raise RuntimeError(f"Conversion coverage failed: converted={len(converted)}, expected={len(expected)}, missing={missing[:10]}")

    output_transformer = args.output_pipeline / "transformer"
    output_transformer.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base_pipeline / "transformer" / "config.json", output_transformer / "config.json")
    save_file(converted, output_transformer / "diffusion_pytorch_model.safetensors", metadata={"format": "pt"})
    for name in ("model_index.json", "scheduler", "tokenizer", "text_encoder", "vae"):
        source = args.base_pipeline / name
        target = args.output_pipeline / name
        if not target.exists():
            target.symlink_to(source)
    print(f"Converted {len(converted)} tensors to {args.output_pipeline}")


if __name__ == "__main__":
    main()
