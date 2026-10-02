#!/usr/bin/env python3
"""Compare SageAttention v2 (INT8 QK / FP8 PV) against torch SDPA on H3-like shapes."""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F


def build_h3_like(s: int, heads: int = 56, dim: int = 128, dtype=torch.bfloat16):
    """Q/K mimic H3 post-QK-RMSNorm+RoPE (unit RMS scaled to sqrt(dim)); V is unnormalised."""
    torch.manual_seed(0)
    q = torch.randn(1, heads, s, dim, device="cuda", dtype=dtype)
    k = torch.randn(1, heads, s, dim, device="cuda", dtype=dtype)
    q = (q / q.norm(dim=-1, keepdim=True) * math.sqrt(dim)).contiguous()
    k = (k / k.norm(dim=-1, keepdim=True) * math.sqrt(dim)).contiguous()
    v = (torch.randn(1, heads, s, dim, device="cuda", dtype=dtype) * 0.5).contiguous()
    return q, k, v


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).square().sum() / b.float().square().sum()).sqrt().item()


def main() -> None:
    from sageattention import sageattn

    scale = 128 ** -0.5
    for s in (4096, 16384):
        q, k, v = build_h3_like(s)
        with torch.inference_mode():
            ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
            out = sageattn(q, k, v, tensor_layout="HND", sm_scale=scale)
            err = rel_l2(out, ref)

            for _ in range(2):
                sageattn(q, k, v, tensor_layout="HND", sm_scale=scale)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                sageattn(q, k, v, tensor_layout="HND", sm_scale=scale)
            torch.cuda.synchronize()
            t1 = time.time()

            for _ in range(2):
                F.scaled_dot_product_attention(q, k, v, scale=scale)
            torch.cuda.synchronize()
            t2 = time.time()
            for _ in range(5):
                F.scaled_dot_product_attention(q, k, v, scale=scale)
            torch.cuda.synchronize()
            t3 = time.time()

        print(f'S={s:6d}  rel_l2={err:.4f}  '
              f'sageattn={1000 * (t1 - t0) / 5:.1f}ms  sdpa={1000 * (t3 - t2) / 5:.1f}ms')


if __name__ == "__main__":
    main()
