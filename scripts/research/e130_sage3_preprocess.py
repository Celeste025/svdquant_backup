"""Experimental Sage3 preprocessing that avoids materializing padded Q/K/V.

The native FP4 K-permute and V-transpose packers already write padded output
from a shorter input. This module exercises that existing ABI; it does not
replace the corrected Sage3 attention core or its numerical recipe.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _group_mean_pad_q(
    Q, QO, QM,
    N: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
    SQB: tl.constexpr, SQH: tl.constexpr, SQN: tl.constexpr, SQD: tl.constexpr,
    SOB: tl.constexpr, SOH: tl.constexpr, SON: tl.constexpr, SOD: tl.constexpr,
    SMB: tl.constexpr, SMH: tl.constexpr, SMG: tl.constexpr, SMD: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)
    t = g * 128 + tl.arange(0, 128)
    d = tl.arange(0, D)
    x = tl.load(Q + b * SQB + h * SQH + t[:, None] * SQN + d[None, :] * SQD,
                mask=t[:, None] < N, other=0)
    mean = tl.sum(x, 0) / 128
    centered = x - mean[None, :]
    tl.store(QO + b * SOB + h * SOH + t[:, None] * SON + d[None, :] * SOD, centered)
    tl.store(QM + b * SMB + h * SMH + g * SMG + d * SMD, mean)


def q_group_mean_with_pad(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, n, d = q.shape
    if d not in (64, 128):
        raise ValueError(f"unsupported head dim: {d}")
    p = triton.cdiv(n, 128) * 128
    out = torch.empty((b, h, p, d), dtype=q.dtype, device=q.device)
    mean = torch.empty((b, h, p // 128, d), dtype=q.dtype, device=q.device)
    _group_mean_pad_q[(b, h, p // 128)](
        q, out, mean, n, p, d, *q.stride(), *out.stride(), *mean.stride(),
        # Match the upstream triton_group_mean reduction tree (default 4
        # warps); 8 warps changes BF16 rounding and some FP4 decisions.
        num_warps=4,
    )
    return out, mean


@triton.jit
def _center_pad_k(
    K, KM, KO,
    N: tl.constexpr, D: tl.constexpr,
    SKB: tl.constexpr, SKH: tl.constexpr, SKN: tl.constexpr, SKD: tl.constexpr,
    SMB: tl.constexpr, SMH: tl.constexpr, SMD: tl.constexpr,
    SOB: tl.constexpr, SOH: tl.constexpr, SON: tl.constexpr, SOD: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile = tl.program_id(2)
    t = tile * 128 + tl.arange(0, 128)
    d = tl.arange(0, D)
    x = tl.load(K + b * SKB + h * SKH + t[:, None] * SKN + d[None, :] * SKD,
                mask=t[:, None] < N, other=0)
    mean = tl.load(KM + b * SMB + h * SMH + d * SMD)
    centered = tl.where(t[:, None] < N, x - mean[None, :], 0)
    tl.store(KO + b * SOB + h * SOH + t[:, None] * SON + d[None, :] * SOD,
             centered)


def k_center_with_pad(k: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    b, h, n, d = k.shape
    if d not in (64, 128):
        raise ValueError(f"unsupported head dim: {d}")
    p = triton.cdiv(n, 128) * 128
    out = torch.empty((b, h, p, d), dtype=k.dtype, device=k.device)
    _center_pad_k[(b, h, p // 128)](
        k, mean, out, n, d, *k.stride(),
        mean.stride(0), mean.stride(1), mean.stride(3), *out.stride(),
        num_warps=4,
    )
    return out


def quant_permute_with_implicit_pad(k: torch.Tensor, api):
    b, h, n, d = k.shape
    p = triton.cdiv(n, 128) * 128
    code = torch.empty((b, h, p, d // 2), device=k.device, dtype=torch.uint8)
    scale = torch.empty((b, h, p, d // 16), device=k.device, dtype=torch.float8_e4m3fn)
    api.fp4quant_cuda.scaled_fp4_quant_permute(k, code, scale, 1)
    return code, scale


def quant_permute_center_with_implicit_pad(k: torch.Tensor, mean: torch.Tensor, api):
    """Center K in place and pack it without a separate K-subtract kernel."""
    b, h, n, d = k.shape
    p = triton.cdiv(n, 128) * 128
    code = torch.empty((b, h, p, d // 2), device=k.device, dtype=torch.uint8)
    scale = torch.empty((b, h, p, d // 16), device=k.device, dtype=torch.float8_e4m3fn)
    api.fp4quant_cuda.scaled_fp4_quant_permute_center(k, mean, code, scale)
    return code, scale


def quant_transpose_with_implicit_pad(v: torch.Tensor, api):
    b, h, n, d = v.shape
    p = triton.cdiv(n, 128) * 128
    code = torch.empty((b, h, d, p // 2), device=v.device, dtype=torch.uint8)
    scale = torch.empty((b, h, d, p // 16), device=v.device, dtype=torch.float8_e4m3fn)
    api.fp4quant_cuda.scaled_fp4_quant_trans(v, code, scale, 1)
    return code, scale


def prepack(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, api,
            *, fuse_q_pad_mean: bool = True, implicit_kv_pad: bool = True,
            fuse_k_center_pack: bool = False, exact_delta: bool = False,
            delta_dtype: torch.dtype = torch.float32,
            fuse_k_center_pad: bool = False,
            mask_padded_delta: bool | None = None):
    """Return official Sage3 packed operands, with optional pad-copy removal."""
    if delta_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"unsupported correction dtype: {delta_dtype}")
    if mask_padded_delta is None:
        mask_padded_delta = bool(getattr(api.fp4attn_cuda,
                                         "delta_tail_mask_in_delta", False))
    n = q.shape[2]
    p = triton.cdiv(n, 128) * 128
    if fuse_k_center_pack and not implicit_kv_pad:
        raise ValueError("fused K center/pack currently requires implicit K/V padding")
    if fuse_k_center_pack and fuse_k_center_pad:
        raise ValueError("choose only one fused K centering strategy")
    if fuse_k_center_pad and not exact_delta:
        raise ValueError("fused K centering/padding requires exact_delta")
    k_mean = k.mean(dim=2, keepdim=True)
    if fuse_k_center_pack:
        kp = quant_permute_center_with_implicit_pad(k, k_mean, api)
    elif fuse_k_center_pad:
        kpad = k_center_with_pad(k, k_mean)
    else:
        k.sub_(k_mean)
    if fuse_q_pad_mean:
        qp, qm = q_group_mean_with_pad(q)
    else:
        qp, qm = api.triton_group_mean(F.pad(q, (0, 0, 0, p - n)).contiguous())
    if implicit_kv_pad:
        if exact_delta:
            # Preserve the official GEMM shape/reduction path. The cheaper
            # unpadded GEMM below differs by a few BF16-rounded elements,
            # which can amplify across many DiT blocks.
            if not fuse_k_center_pad:
                kpad = F.pad(k, (0, 0, 0, p - n)).contiguous()
            delta_s = torch.matmul(qm, kpad.transpose(-2, -1)).to(delta_dtype).contiguous()
        else:
            # The correction has zeros in all padded K columns, as in
            # official preprocess_qkv. Avoid materializing padded K here.
            delta_s = F.pad(torch.matmul(qm, k.transpose(-2, -1)).to(delta_dtype),
                            (0, p - n)).contiguous()
        if not fuse_k_center_pack:
            kp = quant_permute_with_implicit_pad(kpad if fuse_k_center_pad else k, api)
        vp = quant_transpose_with_implicit_pad(v, api)
    else:
        if not fuse_k_center_pad:
            kpad = F.pad(k, (0, 0, 0, p - n)).contiguous()
        vpad = F.pad(v, (0, 0, 0, p - n)).contiguous()
        delta_s = torch.matmul(qm, kpad.transpose(-2, -1)).to(delta_dtype).contiguous()
        kp = api.scale_and_quant_fp4_permute(kpad)
        vp = api.scale_and_quant_fp4_transpose(vpad)
    qpack = api.scale_and_quant_fp4(qp)
    if mask_padded_delta and p > n:
        delta_s[..., n:] = float("-inf")
    return qpack, kp, vp, delta_s, (qp, qm)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, api,
              *, fuse_q_pad_mean: bool = True, implicit_kv_pad: bool = True,
              fuse_k_center_pack: bool = False, exact_delta: bool = False,
              contiguous_output: bool = True,
              delta_dtype: torch.dtype = torch.float32,
              fuse_k_center_pad: bool = False,
              mask_padded_delta: bool | None = None):
    n = q.shape[2]
    qpack, kpack, vpack, delta_s, _ = prepack(
        q, k, v, api, fuse_q_pad_mean=fuse_q_pad_mean,
        implicit_kv_pad=implicit_kv_pad, fuse_k_center_pack=fuse_k_center_pack,
        exact_delta=exact_delta, delta_dtype=delta_dtype,
        fuse_k_center_pad=fuse_k_center_pad,
        mask_padded_delta=mask_padded_delta,
    )
    out = api.blockscaled_fp4_attn(qpack, kpack, vpack, delta_s, n,
                                   is_causal=False, per_block_mean=True,
                                   is_bf16=q.dtype == torch.bfloat16)[0][:, :, :n, :]
    return out.contiguous() if contiguous_output else out
