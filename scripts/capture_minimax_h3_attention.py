#!/usr/bin/env python3
"""Capture one real BF16 MiniMax-H3 main-attention invocation.

The hook is deliberately placed around DiffSynth's existing varlen-attention
helper.  Therefore Q/K/V are captured *after* H3's QK-RMSNorm and RoPE, and
the model's fused-QKV layout is never reproduced by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from minimax_h3_svdquant_common import DATA_ROOT, load_h3_pipeline, read_prompt


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_meta(x: torch.Tensor) -> dict:
    return {"shape": list(x.shape), "dtype": str(x.dtype), "stride": list(x.stride()),
            "numel": x.numel(), "bytes": x.numel() * x.element_size()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-id", type=int, default=1)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--capture-step", type=int, default=0,
                   help="0-based DiT invocation; only the first invocation is captured by default")
    p.add_argument("--min-free-gib", type=float, default=30.0)
    p.add_argument("--reserve-gib", type=float, default=4.0)
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("H3_ATTN_CAPTURE_DIR", DATA_ROOT / "research/E089_h3_attention")))
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "results/research/E089_h3_attention/manifest.json")
    args = p.parse_args()
    if args.height % 32 or args.width % 32 or (args.frames - 5) % 17:
        raise ValueError("height/width must be multiples of 32 and frames must satisfy 17n+5")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    free, total = torch.cuda.mem_get_info("cuda")
    preflight = {"free_gib": free / 2**30, "total_gib": total / 2**30,
                 "required_free_gib": args.min_free_gib, "device": torch.cuda.get_device_name()}
    if preflight["free_gib"] < args.min_free_gib:
        args.manifest.write_text(json.dumps({"state": "deferred_no_gpu_capacity", "preflight": preflight}, indent=2) + "\n")
        raise SystemExit("insufficient free GPU memory; wrote deferred manifest")

    row = read_prompt(args.prompt_id)
    comfy = importlib.import_module("diffsynth.models.minimax_h3_dit_comfy")
    original_attention = comfy._sdpa_varlen_attention
    pipe = load_h3_pipeline(full=True, reserve_gib=args.reserve_gib)
    target = pipe.dit.blocks[0].attn
    captured: dict[str, object] = {}
    state = {"active": False, "dit_calls": 0, "capture": False}

    def dit_pre(_module, _args, _kwargs):
        state["capture"] = state["dit_calls"] == args.capture_step
        state["dit_calls"] += 1

    def attn_pre(_module, _args, _kwargs):
        state["active"] = bool(state["capture"] and "q" not in captured)

    def attn_post(_module, _args, _kwargs, output):
        state["active"] = False
        return output

    def qkv_hook(_module, inputs, output):
        if state["active"]:
            captured["qkv_proj_input"] = inputs[0].detach().cpu().contiguous()
            captured["qkv_proj_output"] = output.detach().cpu().contiguous()

    def out_hook(_module, inputs, output):
        if state["active"]:
            captured["out_proj_input"] = inputs[0].detach().cpu().contiguous()
            captured["out_proj_output"] = output.detach().cpu().contiguous()

    def wrapped_attention(q, k, v, cu_seqlens, softmax_scale):
        result = original_attention(q, k, v, cu_seqlens, softmax_scale)
        if state["active"]:
            captured.update({
                "q": q.detach().cpu().contiguous(), "k": k.detach().cpu().contiguous(),
                "v": v.detach().cpu().contiguous(), "attention_output": result.detach().cpu().contiguous(),
                "cu_seqlens": cu_seqlens.detach().cpu().contiguous(), "softmax_scale": float(softmax_scale),
            })
        return result

    handles = [pipe.dit.register_forward_pre_hook(dit_pre, with_kwargs=True),
               target.register_forward_pre_hook(attn_pre, with_kwargs=True),
               target.register_forward_hook(attn_post, with_kwargs=True),
               target.qkv_proj.register_forward_hook(qkv_hook), target.out_proj.register_forward_hook(out_hook)]
    comfy._sdpa_varlen_attention = wrapped_attention
    try:
        pipe(prompt=row["prompt"], seed=int(row["seed"]), height=args.height, width=args.width,
             num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0,
             rand_device="cpu", tiled=True)
    finally:
        comfy._sdpa_varlen_attention = original_attention
        for handle in handles:
            handle.remove()

    required = {"qkv_proj_input", "qkv_proj_output", "q", "k", "v", "attention_output",
                "out_proj_input", "out_proj_output", "cu_seqlens", "softmax_scale"}
    missing = required - captured.keys()
    if missing:
        raise RuntimeError(f"target attention was not captured: {sorted(missing)}")
    cu = captured["cu_seqlens"]
    valid_lengths = [int(b - a) for a, b in zip(cu[:-1], cu[1:])]
    payload = {"tensors": captured, "metadata": {
        "model": type(pipe.dit).__name__, "block": 0, "prompt_id": args.prompt_id,
        "seed": int(row["seed"]), "capture_step": args.capture_step, "height": args.height,
        "width": args.width, "frames": args.frames, "steps": args.steps,
        "valid_lengths": valid_lengths, "preflight": preflight, "torch": torch.__version__,
        "created_unix": time.time(),
    }}
    # Needed for the requested post-o-projection error, without retaining the
    # full DiT checkpoint in the offline analysis process.
    payload["tensors"]["out_proj_weight"] = target.out_proj.weight.detach().cpu().contiguous()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    output = args.data_dir / f"block0_prompt{args.prompt_id}_step{args.capture_step}.pt"
    torch.save(payload, output)
    manifest = {"state": "complete", "capture": str(output), "sha256": sha256(output),
                "metadata": payload["metadata"],
                "tensors": {k: tensor_meta(v) for k, v in captured.items() if isinstance(v, torch.Tensor)}}
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
