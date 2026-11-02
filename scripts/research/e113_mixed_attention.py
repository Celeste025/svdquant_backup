"""H3 mixed P/V reference: official Sage3 preprocessing plus MixAttention Triton core.

Q/K centering and correction use the same official Sage3 preprocessing as the
pure-Sage3 arm. Only the P/V tile precision decision differs. This remains a
BF16-matmul numerical reference, not a native mixed-FP4/FP8 speed kernel.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
MIX = ROOT / "third_party/MixAttention"
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
sys.path.insert(0, str(MIX))
sys.path.insert(0, str(SAGE))

from pv_hadamard.quant import nvfp4_qdq
from triton_adaptive_pv import _adaptive_pv_kernel
from sageattn3.api import preprocess_qkv


def mixed_sage3_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          *, threshold: float = 0.10) -> torch.Tensor:
    """Inputs/output `[1, H, L, 128]` BF16; noncausal, per-128 Q mean."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("expected matching [1,H,L,D] QKV")
    if q.dtype != torch.bfloat16 or not q.is_cuda or q.shape[-1] != 128:
        raise ValueError("expected CUDA BF16 D=128")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0,1)")
    _, heads, length, dim = q.shape
    qpp, kpp, vpp, delta_s = preprocess_qkv(q, k.clone(), v, per_block_mean=True)
    padded = qpp.shape[2]
    assert padded == kpp.shape[2] == vpp.shape[2] == math.ceil(length / 128) * 128
    assert delta_s.shape == (1, heads, padded // 128, padded)
    qdq = nvfp4_qdq(qpp[0].float()).contiguous()
    kdq = nvfp4_qdq(kpp[0].float()).contiguous()
    vf = vpp[0].float().contiguous()
    ds = delta_s[0].float().contiguous()
    if padded != length:
        ds[:, :, length:] = -float("inf")
    scale = dim ** -0.5
    ds.mul_(scale)
    out = torch.empty_like(qdq)
    _adaptive_pv_kernel[(triton.cdiv(padded, 128), heads)](
        qdq, kdq, vf, ds, out,
        *qdq.stride(), *kdq.stride(), *vf.stride(), *ds.stride(), *out.stride(),
        L=padded, K=padded, BLOCK_Q=128, GROUP=16, NGROUPS=8, HEAD_DIM=128,
        scale=scale, threshold_sq=threshold * threshold)
    result = out[:, :length].unsqueeze(0).to(torch.bfloat16).contiguous()
    assert result.shape == q.shape
    return result
