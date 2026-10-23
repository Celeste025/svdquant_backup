"""Experimental native and split mixed Sage3 attention for SM120.

Both entry points accept BF16 Q/K/V in [1, H, L, 128] layout. They use
noncausal attention, per-128-token Q centering, and threshold 0.10 by default.
Build the corrected FP4, native mixed, sparse V router, and LSE extensions
before importing this module (see results/research/E120/README.md).
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                str(ROOT / "third_party/sage3_native_mixed"),
                str(Path(__file__).resolve().parent)]

import fp4attn_lse_cuda
import fp4attn_mixed_maskfix_cuda
import sage3_mixed_router_sparse_cuda
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)

from e120_split_native_mixed import (dequant_qk4, split_selected_fp8_pv,
                                     zero_selected_v4)


@triton.jit
def _compact_route_all(ROUTE, IDS, COUNTS, TILES: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Compact every tile, including indices >=128 in long H3 sequences."""
    head = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    chosen = tl.load(ROUTE + head * TILES + idx,
                     mask=idx < TILES, other=0).to(tl.int32)
    pos = tl.cumsum(chosen) - 1
    tl.store(IDS + head * TILES + pos, idx,
             mask=(idx < TILES) & (chosen != 0))
    tl.store(COUNTS + head, tl.sum(chosen))


def _pack(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if (q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or
            q.shape[0] != 1 or q.shape[-1] != 128 or
            q.dtype != torch.bfloat16 or not q.is_cuda):
        raise ValueError("expected matching CUDA BF16 Q/K/V of shape [1,H,L,128]")
    length = q.shape[2]
    qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
    q4 = scale_and_quant_fp4(qp)
    k4 = scale_and_quant_fp4_permute(kp)
    v4 = scale_and_quant_fp4_transpose(vp)
    packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
              ds, length, None, 128 ** -0.5, False, True, True)
    return length, qp, kp, vp, q4, k4, v4, ds, packet


@torch.inference_mode()
def native_mixed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           *, threshold: float = 0.10) -> torch.Tensor:
    """Original corrected native mixed core, including dynamic packing."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0,1)")
    length, _, _, vp, _, _, _, _, packet = _pack(q, k, v)
    v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(
        vp, threshold)
    return fp4attn_mixed_maskfix_cuda.fwd_mixed(
        *packet, v8, s8, route)[0][:, :, :length].contiguous()


@torch.inference_mode()
def split_mixed_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          *, threshold: float = 0.10) -> torch.Tensor:
    """Spill-free FP4 core plus sparse P8×V8 correction; experimental."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0,1)")
    length, qp, kp, vp, q4, k4, v4, ds, packet = _pack(q, k, v)
    heads = q.shape[1]
    tiles = vp.shape[2] // 128
    v8, s8, route, _ = sage3_mixed_router_sparse_cuda.route_and_pack_v(
        vp, threshold)
    ids = torch.empty_like(route, dtype=torch.int32)
    counts = torch.empty((1, heads), device=q.device, dtype=torch.int32)
    qdec = torch.empty((1, heads, qp.shape[2], 128),
                       device=q.device, dtype=torch.float16)
    kdec = torch.empty((1, heads, kp.shape[2], 128),
                       device=q.device, dtype=torch.float16)
    zero_selected_v4[(tiles, heads)](v4[0], route, heads, vp.shape[2], tiles)
    _compact_route_all[(heads,)](route, ids, counts, tiles,
                                 triton.next_power_of_2(tiles))
    dequant_qk4[(qp.shape[2] // 32, heads)](
        q4[0], q4[1], route, qdec, qp.shape[2], tiles, False)
    dequant_qk4[(kp.shape[2] // 32, heads)](
        k4[0], k4[1], route, kdec, kp.shape[2], tiles, True)
    base_full, lse_full = fp4attn_lse_cuda.fwd(*packet)[:2]
    base = base_full[:, :, :length].contiguous()
    lse = lse_full[:, :, :length].contiguous()
    out = torch.empty_like(base)
    split_selected_fp8_pv[(triton.cdiv(length, 64), heads)](
        q4[0], k4[0], q4[1], k4[1], qdec, kdec, ds, v8, s8,
        ids, counts, lse, base, out, length, qp.shape[2], kp.shape[2],
        tiles, 64, True, num_warps=8, num_stages=2)
    return out
