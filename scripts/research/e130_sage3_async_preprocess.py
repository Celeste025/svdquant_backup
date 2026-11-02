#!/usr/bin/env python3
"""Experimental exact Sage3 prepack schedule using independent CUDA streams.

The arithmetic and kernels match e130_sage3_preprocess's fused K-pad path.
Only independent Q, K-pack, and V-pack work is overlapped with the correction
GEMM. This is an opt-in experiment, not the default inference route.
"""
from __future__ import annotations

import torch
import triton

from e130_sage3_preprocess import (
    k_center_with_pad,
    q_group_mean_with_pad,
    quant_permute_with_implicit_pad,
    quant_transpose_with_implicit_pad,
)


class AsyncSage3Prepacker:
    def __init__(self) -> None:
        self.q_stream = torch.cuda.Stream()
        self.k_stream = torch.cuda.Stream()
        self.v_stream = torch.cuda.Stream()

    @staticmethod
    def _record_pair(pair: tuple[torch.Tensor, torch.Tensor], stream: torch.cuda.Stream) -> None:
        for tensor in pair:
            tensor.record_stream(stream)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, api,
                  *, contiguous_output: bool = True) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
            raise ValueError("expected matched BF16 HND Q/K/V with head dim 128")
        if q.dtype != torch.bfloat16:
            raise ValueError("experimental async path requires BF16")
        if getattr(api.fp4attn_cuda, "delta_s_dtype", None) != "bf16":
            raise ValueError("experimental async path requires BF16 delta extension")
        current = torch.cuda.current_stream(q.device)
        n = q.shape[2]
        p = triton.cdiv(n, 128) * 128

        self.q_stream.wait_stream(current)
        self.v_stream.wait_stream(current)
        q.record_stream(self.q_stream)
        v.record_stream(self.v_stream)
        with torch.cuda.stream(self.q_stream):
            qp, qm = q_group_mean_with_pad(q)
            q_mean_ready = torch.cuda.Event()
            q_mean_ready.record()
            qpack = api.scale_and_quant_fp4(qp)
        with torch.cuda.stream(self.v_stream):
            vpack = quant_transpose_with_implicit_pad(v, api)

        k_mean = k.mean(dim=2, keepdim=True)
        kpad = k_center_with_pad(k, k_mean)
        self.k_stream.wait_stream(current)
        kpad.record_stream(self.k_stream)
        with torch.cuda.stream(self.k_stream):
            kpack = quant_permute_with_implicit_pad(kpad, api)

        current.wait_event(q_mean_ready)
        qm.record_stream(current)
        delta_s = torch.matmul(qm, kpad.transpose(-2, -1)).contiguous()
        if p > n and getattr(api.fp4attn_cuda, "delta_tail_mask_in_delta", False):
            delta_s[..., n:] = float("-inf")

        current.wait_stream(self.q_stream)
        current.wait_stream(self.k_stream)
        current.wait_stream(self.v_stream)
        self._record_pair(qpack, current)
        self._record_pair(kpack, current)
        self._record_pair(vpack, current)
        out = api.blockscaled_fp4_attn(qpack, kpack, vpack, delta_s, n,
                                       is_causal=False, per_block_mean=True,
                                       is_bf16=True)[0][:, :, :n, :]
        return out.contiguous() if contiguous_output else out
