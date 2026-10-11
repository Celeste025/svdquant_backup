#!/usr/bin/env python3
"""Export one real MiniMax-H3 block replay for native-GEMM benchmarking.

The saved tensors are deliberately small in scope: one cached denoising call,
the exact input to a selected DiT block, the real inputs to its four quantized
linears, and its BF16 output.  The loader uses DiffSynth disk offload, so this
can run beside services that leave only a few GiB of GPU memory free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import (
    load_h3_pipeline,
    raw_call_to_block0,
    run_block,
    tree_cpu,
    tree_device,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-call", type=Path, required=True)
    p.add_argument("--target-block", type=int, default=25)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vram-limit-gib", type=float, default=4.5)
    return p.parse_args()


def main() -> None:
    a = args()
    if not 0 <= a.target_block < 50:
        raise ValueError("target block must be in [0, 49]")
    if not a.cache_call.is_file():
        raise FileNotFoundError(a.cache_call)

    raw = torch.load(a.cache_call, map_location="cpu", weights_only=False)
    pipe = load_h3_pipeline(
        full=False,
        reserve_gib=max(0.0, 72.0 - a.vram_limit_gib),
        vram_limit_gib=a.vram_limit_gib,
    )
    pipe.load_models_to_device(["dit"])
    dit = pipe.dit
    hidden, kwargs = raw_call_to_block0(dit, raw)
    for block_id in range(a.target_block):
        print(f"replay predecessor {block_id}/{a.target_block - 1}", flush=True)
        hidden = run_block(dit.blocks[block_id], [(hidden, kwargs)])[0]
        # The pipeline manager leaves every touched VRAMLinear prepared.  This
        # replay will never revisit predecessor blocks, so explicitly return
        # their parameters to the model's disk-offload shell.  Without this,
        # a long prefix silently accumulates ~one block of weights per step.
        for module in dit.blocks[block_id].modules():
            offload = getattr(module, "offload", None)
            if callable(offload):
                offload()
        torch.cuda.empty_cache()

    block = dit.blocks[a.target_block]
    captures: dict[str, torch.Tensor] = {}
    handles = []
    def capture(key: str):
        def hook(_module, inputs):
            captures.setdefault(key, inputs[0].detach().cpu())
            return None
        return hook
    for name in ("qkv_proj", "out_proj"):
        linear = getattr(block.attn, name)
        handles.append(linear.register_forward_pre_hook(capture(f"attn.{name}")))
    for name in ("fc1", "fc2"):
        linear = getattr(block.mlp, name)
        handles.append(linear.register_forward_pre_hook(capture(f"mlp.{name}")))
    try:
        with torch.inference_mode():
            reference = block(hidden.to("cuda"), **tree_device(kwargs, "cuda")).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    expected = {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"}
    if captures.keys() != expected:
        raise RuntimeError(f"missing linear captures: got {sorted(captures)}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "h3-native-bench-sample-v1",
        "cache_call": str(a.cache_call),
        "target_block": a.target_block,
        "block_input": hidden.cpu(),
        "block_kwargs": tree_cpu(kwargs),
        "linear_inputs": captures,
        "block_reference": reference,
    }
    torch.save(payload, a.output)
    manifest = {
        "format": payload["format"],
        "cache_call": str(a.cache_call),
        "target_block": a.target_block,
        "block_input_shape": list(hidden.shape),
        "block_reference_shape": list(reference.shape),
        "linear_input_shapes": {key: list(value.shape) for key, value in captures.items()},
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "output": str(a.output),
    }
    a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
