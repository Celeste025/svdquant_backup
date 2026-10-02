"""SageAttention3 integration for the DiffSynth MiniMax-H3 path.

The H3 runtime uses ``MiniMaxH3DiTComfyPruned``, whose attention forward is
``_comfy_attention_forward``; every attention call (main blocks and token
refiner) funnels through the module-global ``_sdpa_varlen_attention`` of
``diffsynth.models.minimax_h3_dit_comfy``.  Patching that single function
swaps the backend with no risk of re-implementing the qkv unpack layout.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch


def _load_sageattn3():
    try:
        from sageattn3 import sageattn3_blackwell
    except ImportError as exc:
        raise RuntimeError(
            "SageAttention3 is not installed in this Python environment. "
            "Compile app_source/SageAttention/sageattention3_blackwell first."
        ) from exc
    return sageattn3_blackwell


def _sageattn3_varlen_attention(q, k, v, cu_seqlens, softmax_scale):
    sageattn3_blackwell = _load_sageattn3()
    head_dim = q.shape[-1]
    if softmax_scale is not None and abs(softmax_scale - head_dim**-0.5) > 1e-12:
        raise RuntimeError(
            f"sageattn3 hardcodes 1/sqrt(head_dim)={head_dim**-0.5}, "
            f"but the caller passed softmax_scale={softmax_scale}"
        )
    output = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        segment = slice(start, stop)
        segment_output = sageattn3_blackwell(
            q[segment].transpose(0, 1).unsqueeze(0),
            k[segment].transpose(0, 1).unsqueeze(0),
            v[segment].transpose(0, 1).unsqueeze(0),
            is_causal=False,
        )
        output[segment] = segment_output.squeeze(0).transpose(0, 1)
    return output


@dataclass
class SageAttn3Runtime:
    patched_modules: list

    def remove(self) -> None:
        for module in self.patched_modules:
            module._sdpa_varlen_attention = module._sageattn3_original_attention
            del module._sageattn3_original_attention


def install_sageattn3(dit: torch.nn.Module) -> SageAttn3Runtime:
    blocks = getattr(dit, "blocks", None)
    if blocks is None or len(blocks) != 50:
        raise RuntimeError("Expected MiniMax-H3 DiT with exactly 50 main blocks")
    _load_sageattn3()
    comfy = importlib.import_module("diffsynth.models.minimax_h3_dit_comfy")
    if getattr(comfy, "_sageattn3_original_attention", None) is not None:
        raise RuntimeError("minimax_h3_dit_comfy is already patched with SageAttention3")
    comfy._sageattn3_original_attention = comfy._sdpa_varlen_attention
    comfy._sdpa_varlen_attention = _sageattn3_varlen_attention
    return SageAttn3Runtime([comfy])
