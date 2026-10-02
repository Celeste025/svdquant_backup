#!/usr/bin/env python3
"""Localize where SageAttention breaks the H3 SVDQuant denoising loop.

Runs a two-step generation with instrumented attention call sites and reports
per-call input/output magnitude and NaN counts for both backends.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch

from minimax_h3_svdquant_common import load_h3_pipeline
from minimax_h3_vbench51_worker import SETTINGS, apply_svdquant

import diffsynth.models.minimax_h3_dit as h3_dit

attention_module = importlib.import_module("diffsynth.core.attention.attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "sage_attention"), required=True)
    parser.add_argument("--variant", choices=("bf16", "svdquant"), default="svdquant")
    parser.add_argument("--state", type=Path,
                        default=Path("/home/admin/workspace/aop_lab/app_data/artifacts/variants/nvfp4-g10-r64/quant_state.pt"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3"
                                     "/metadata/h3_vbench2_100/manifest.json"))
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--dump-short", type=Path,
                        help="save q/k/v of segments shorter than 256 tokens for offline replay")
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def stats(tag: str, tensor: torch.Tensor) -> dict:
    t = tensor.detach().float()
    return {"tag": tag, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()),
            "amax": float(t.abs().amax()) if t.numel() else 0.0,
            "rms": float(t.pow(2).mean().sqrt()) if t.numel() else 0.0}


def main() -> None:
    args = parse_args()
    attention_module.ATTENTION_IMPLEMENTATION = args.backend
    records: list[dict] = []
    call_index = {"n": 0}
    dumps: dict[int, int] = {}

    def wrapped(q, k, v, *a, **kw):
        out = h3_dit_original(q, k, v, *a, **kw)
        if args.dump_short is not None and int(q.shape[-2]) < 256 and call_index["n"] not in dumps:
            dumps[call_index["n"]] = call_index["n"]
            torch.save({"call": call_index["n"], "q": q.detach().cpu(), "k": k.detach().cpu(),
                        "v": v.detach().cpu(), "out": out.detach().cpu(),
                        "strides": (q.stride(), k.stride(), v.stride())},
                       args.dump_short.with_name(f"{args.dump_short.name}_call{call_index['n']}.pt"))
        if call_index["n"] < args.max_calls:
            records.append({
                "impl": attention_module.ATTENTION_IMPLEMENTATION,
                "call": call_index["n"],
                "seg_len": int(q.shape[-2]),
                "q": stats("q", q), "k": stats("k", k), "v": stats("v", v),
                "out": stats("out", out),
                "q_contig": q.is_contiguous(), "k_contig": k.is_contiguous(),
                "v_contig": v.is_contiguous(),
            })
        call_index["n"] += 1
        return out

    h3_dit_original = h3_dit.attention_forward
    h3_dit.attention_forward = wrapped

    case = json.loads(args.manifest.read_text())["cases"][args.case_index]
    pipe = load_h3_pipeline(full=True, reserve_gib=args.reserve_gib)
    if args.variant == "svdquant":
        pipe.load_models_to_device(["dit"])
        apply_svdquant(pipe.dit, args.state)
    step_records: list[dict] = []

    def step_hook(_module, call_args, call_kwargs):
        x = call_args[0] if call_args else call_kwargs["x"]
        cu = call_kwargs["packed_seq_params"]["cu_seqlens_q"]
        step_records.append({"dit_call": len(step_records), **stats("dit_input", x),
                             "cu_seqlens": [int(v) for v in cu.reshape(-1).tolist()]})

    handle = pipe.dit.register_forward_pre_hook(step_hook, with_kwargs=True)
    try:
        video = pipe(prompt=case["prompt"], seed=int(SETTINGS["seed"]), height=SETTINGS["height"],
                     width=SETTINGS["width"], num_frames=SETTINGS["frames"],
                     num_inference_steps=args.steps, cfg_scale=1.0, tiled=True)
    finally:
        handle.remove()

    out = video[0] if isinstance(video, (list, tuple)) else video
    payload = {
        "backend": args.backend, "variant": args.variant, "steps": args.steps,
        "attention_calls": call_index["n"], "attention_records": records,
        "dit_inputs": step_records,
        "video_stats": stats("video", out) if isinstance(out, torch.Tensor) else str(type(out)),
    }
    args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
