#!/usr/bin/env python3
"""Sanity-check Wan attention struct mapping before PTQ.

Fails if any blocks.*.attn2 is not treated as cross-attn (to_k/to_v on add path).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
WAN_PATH = os.environ.get(
    "WAN_MODEL_PATH",
    "/ssd/2/yuzhibo.yzh_data/Wan2.1-T2V-1.3B-Diffusers",
)


def main() -> int:
    import torch
    from diffusers import WanTransformer3DModel

    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct

    print(f"Loading transformer from {WAN_PATH} ...")
    model = WanTransformer3DModel.from_pretrained(
        WAN_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    struct = DiffusionModelStruct.construct(model)

    n_blocks = 0
    errors: list[str] = []
    for block in struct.iter_transformer_block_structs():
        n_blocks += 1
        if len(block.attn_structs) != 2:
            errors.append(f"{block.name}: expected 2 attns, got {len(block.attn_structs)}")
            continue
        attn1, attn2 = block.attn_structs
        print(
            f"{block.name}: "
            f"attn1 self={attn1.is_self_attn()} cross={attn1.is_cross_attn()} "
            f"qkv={[m is not None for m in (attn1.q_proj, attn1.k_proj, attn1.v_proj)]} "
            f"add_kv={[m is not None for m in (attn1.add_k_proj, attn1.add_v_proj)]}; "
            f"attn2 self={attn2.is_self_attn()} cross={attn2.is_cross_attn()} "
            f"qkv={[m is not None for m in (attn2.q_proj, attn2.k_proj, attn2.v_proj)]} "
            f"add_kv={[m is not None for m in (attn2.add_k_proj, attn2.add_v_proj)]} "
            f"norm_type={block.norm_type}"
        )
        if not attn1.is_self_attn() or attn1.is_cross_attn():
            errors.append(f"{attn1.name}: expected self-attn (q/k/v on qkv path)")
        if attn1.k_proj is None or attn1.v_proj is None:
            errors.append(f"{attn1.name}: missing k/v on self-attn qkv path")
        if not attn2.is_cross_attn() or attn2.is_self_attn():
            errors.append(f"{attn2.name}: expected cross-attn (k/v on add path)")
        if attn2.k_proj is not None or attn2.v_proj is not None:
            errors.append(f"{attn2.name}: k/v must not be on qkv path")
        if attn2.add_k_proj is None or attn2.add_v_proj is None:
            errors.append(f"{attn2.name}: missing add_k/add_v (= to_k/to_v)")
        if attn2.add_k_proj is not attn2.module.to_k or attn2.add_v_proj is not attn2.module.to_v:
            errors.append(f"{attn2.name}: add_k/v must be module.to_k/to_v")
        if block.norm_type != "ada_norm_single":
            errors.append(f"{block.name}: norm_type={block.norm_type!r}, expected ada_norm_single")

    if n_blocks == 0:
        errors.append("no transformer blocks found")

    if errors:
        print("FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"OK: {n_blocks} blocks; all attn1=self, attn2=cross (add_k/v=to_k/to_v)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
