"""SageAttention2++ integration for the DiffSynth MiniMax-H3 path.

DiffSynth already routes H3 attention through ``attention_forward`` whenever
``DIFFSYNTH_ATTENTION_IMPLEMENTATION`` selects a backend, and the H3 blocks call
it once per ``cu_seqlens`` segment.  Two adjustments are needed for H3:

* ``ATTENTION_IMPLEMENTATION`` is latched at import time, so the backend is
  forced through the module global instead of the environment variable.
* The sm_120 CUDA kernels derive the value quantization scale from the block
  amax and divide by it, which turns an exactly-zero value column into NaN.
  H3's alignment-padding segment reaches SageAttention with such a column once
  the qkv projection is SVDQuant-quantized (bf16 keeps a small non-zero value),
  and a single NaN then poisons every SVDQuant activation via its global
  NVFP4 tensor scale.  Nudging ``v`` by a sub-epsilon constant keeps the scale
  finite without perturbing any representable bf16 value.
"""
from __future__ import annotations

import importlib

VALUE_EPS = 1e-8


def install_sageattn2() -> dict[str, object]:
    attention = importlib.import_module("diffsynth.core.attention.attention")
    if not attention.SAGE_ATTN_AVAILABLE:
        raise RuntimeError("sageattention is not importable in this environment")
    attention.ATTENTION_IMPLEMENTATION = "sage_attention"
    guard = getattr(attention.sage_attention, "_h3_value_guard", None)
    if guard is None:
        original = attention.sage_attention

        def guarded(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d",
                    out_pattern="b n s d", dims=None, scale=None):
            return original(q, k, v + VALUE_EPS, q_pattern, k_pattern, v_pattern,
                            out_pattern, dims, scale=scale)

        guarded._h3_value_guard = VALUE_EPS
        attention.sage_attention = guarded
        guard = VALUE_EPS
    return {"implementation": attention.ATTENTION_IMPLEMENTATION, "value_eps": guard}
