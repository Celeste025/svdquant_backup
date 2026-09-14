#!/usr/bin/env python3
"""Sanity: wrap_wan_gated(attn1/ffn) matches y * gate from Wan block math."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")

import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "third_party", "deepcompressor"))

from deepcompressor.app.diffusion.quant.utils import wrap_wan_gated  # noqa: E402

WAN_PATH = os.environ.get("WAN_MODEL_PATH", "/ssd/2/yuzhibo.yzh_data/Wan2.1-T2V-1.3B-Diffusers")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    block: WanTransformerBlock = pipe.transformer.blocks[0]
    block.eval()

    B, T, C = 2, 16, block.attn1.to_q.in_features
    hs = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)
    enc = torch.randn(B, 32, C, device=device, dtype=torch.bfloat16)
    temb = torch.randn(B, 6, C, device=device, dtype=torch.float32)
    # rotary: match rope output roughly — use block via a short forward hook
    # Build a simple complex-compatible rotary of shape matching processor expectation.
    # Wan apply_rotary expects freqs broadcastable with query unflattened.
    # Easiest: run one real block forward to get shapes, but for unit test call attn without rope.
    rotary = None

    with torch.inference_mode():
        gate_msa = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[2]
        c_gate = (block.scale_shift_table + temb.float()).chunk(6, dim=1)[5]

        # attn1 without rope (rotary=None is allowed)
        y_attn = block.attn1(hidden_states=hs)
        wrapped_attn = wrap_wan_gated(block.attn1, block, "msa")
        try:
            wrapped_attn(hidden_states=hs)
            print("FAIL: expected KeyError when temb missing")
            return 1
        except KeyError as e:
            assert "temb" in str(e)
            print("temb required in eval kwargs: OK", flush=True)

        y_wrap = wrapped_attn(hidden_states=hs, temb=temb)
        ref = y_attn * gate_msa.type_as(y_attn)
        err_attn = (y_wrap - ref).float().abs().max().item()

        y_ffn = block.ffn(hs)
        wrapped_ffn = wrap_wan_gated(block.ffn, block, "ffn")
        y_wrap_f = wrapped_ffn(hs, temb=temb)
        ref_f = y_ffn * c_gate.type_as(y_ffn)
        err_ffn = (y_wrap_f - ref_f).float().abs().max().item()

    print(f"attn1 gated max_abs_err={err_attn:.3e}")
    print(f"ffn   gated max_abs_err={err_ffn:.3e}")
    # bfloat16 mul can be slightly noisier
    ok = err_attn < 1e-2 and err_ffn < 1e-2
    if not ok:
        print("FAIL")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
