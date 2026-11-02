"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda


@triton.jit
def group_mean_kernel(
    q_ptr,          
    q_out_ptr,      
    qm_out_ptr,     
    B, H, L, D: tl.constexpr,    
    stride_qb, stride_qh, stride_ql, stride_qd,  
    stride_qmb, stride_qmh, stride_qml, stride_qmd,  
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + tl.arange(0, D)[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets)
    
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + tl.arange(0, D) * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype) 
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE
    )
    return q_out, qm


@triton.jit
def _e130_group_mean_pad_q(
    Q, QO, QM, N: tl.constexpr, D: tl.constexpr,
    SQB: tl.constexpr, SQH: tl.constexpr, SQN: tl.constexpr, SQD: tl.constexpr,
    SOB: tl.constexpr, SOH: tl.constexpr, SON: tl.constexpr, SOD: tl.constexpr,
    SMB: tl.constexpr, SMH: tl.constexpr, SMG: tl.constexpr, SMD: tl.constexpr,
):
    b, h, g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = g * 128 + tl.arange(0, 128)
    d = tl.arange(0, D)
    x = tl.load(Q + b * SQB + h * SQH + t[:, None] * SQN + d[None, :] * SQD,
                mask=t[:, None] < N, other=0)
    mean = tl.sum(x, 0) / 128
    tl.store(QO + b * SOB + h * SOH + t[:, None] * SON + d[None, :] * SOD,
             x - mean[None, :])
    tl.store(QM + b * SMB + h * SMH + g * SMG + d * SMD, mean)


@triton.jit
def _e130_group_mean_q_only(
    Q, QM, N: tl.constexpr, D: tl.constexpr,
    SQB: tl.constexpr, SQH: tl.constexpr, SQN: tl.constexpr, SQD: tl.constexpr,
    SMB: tl.constexpr, SMH: tl.constexpr, SMG: tl.constexpr, SMD: tl.constexpr,
):
    b, h, g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = g * 128 + tl.arange(0, 128)
    d = tl.arange(0, D)
    x = tl.load(Q + b * SQB + h * SQH + t[:, None] * SQN + d[None, :] * SQD,
                mask=t[:, None] < N, other=0)
    mean = tl.sum(x, 0) / 128
    tl.store(QM + b * SMB + h * SMH + g * SMG + d * SMD, mean)


@triton.jit
def _e130_center_pad_k(
    K, KM, KO, N: tl.constexpr, D: tl.constexpr,
    SKB: tl.constexpr, SKH: tl.constexpr, SKN: tl.constexpr, SKD: tl.constexpr,
    SMB: tl.constexpr, SMH: tl.constexpr, SMD: tl.constexpr,
    SOB: tl.constexpr, SOH: tl.constexpr, SON: tl.constexpr, SOD: tl.constexpr,
):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = tile * 128 + tl.arange(0, 128)
    d = tl.arange(0, D)
    x = tl.load(K + b * SKB + h * SKH + t[:, None] * SKN + d[None, :] * SKD,
                mask=t[:, None] < N, other=0)
    mean = tl.load(KM + b * SMB + h * SMH + d * SMD)
    centered = tl.where(t[:, None] < N, x - mean[None, :], 0)
    tl.store(KO + b * SOB + h * SOH + t[:, None] * SON + d[None, :] * SOD,
             centered)


@triton.jit
def _e130_tiled_delta_kernel(QM, K, O, M: tl.constexpr, N: tl.constexpr,
                             SQH: tl.constexpr, SQM: tl.constexpr,
                             SKH: tl.constexpr, SKN: tl.constexpr,
                             SOH: tl.constexpr, SOM: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for k0 in range(tl.cdiv(128, BK)):
        d = k0 * BK + kk
        a = tl.load(QM + h * SQH + m[:, None] * SQM + d[None, :],
                    mask=m[:, None] < M, other=0)
        b = tl.load(K + h * SKH + n[None, :] * SKN + d[:, None],
                    mask=n[None, :] < N, other=0)
        acc = tl.dot(a, b, acc)
    tl.store(O + h * SOH + m[:, None] * SOM + n[None, :], acc,
             mask=(m[:, None] < M) & (n[None, :] < N))


def _e130_tiled_delta(qm: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
    """BF16 correction GEMM for validated long-sequence H3 paths."""
    b, h, g, d = qm.shape
    if kp.shape[:2] != (b, h) or kp.shape[-1] != d or d != 128:
        raise ValueError("unexpected correction GEMM shape")
    if not qm.is_contiguous() or not kp.is_contiguous():
        raise ValueError("correction GEMM requires contiguous operands")
    p = kp.shape[2]
    out = torch.empty((b, h, g, p), device=qm.device, dtype=torch.bfloat16)
    qflat = qm.view(b * h, g, d)
    kflat = kp.view(b * h, p, d)
    oflat = out.view(b * h, g, p)
    bm, bn, bk = 32, 128, 32
    _e130_tiled_delta_kernel[(triton.cdiv(g, bm), triton.cdiv(p, bn), b * h)](
        qflat, kflat, oflat, g, p,
        qflat.stride(0), qflat.stride(1),
        kflat.stride(0), kflat.stride(1),
        oflat.stride(0), oflat.stride(1), bm, bn, bk,
        num_warps=4, num_stages=3)
    return out


@triton.jit
def _e130_raw_k_delta_kernel(
    QM, K, KM, O, M: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    SKB: tl.constexpr, SKH: tl.constexpr, SKN: tl.constexpr, SKD: tl.constexpr,
    SQH: tl.constexpr, SQM: tl.constexpr, SMH: tl.constexpr,
    SOH: tl.constexpr, SOM: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    bh = tl.program_id(2)
    b, h = bh // H, bh % H
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for k0 in range(tl.cdiv(128, BK)):
        d = k0 * BK + kk
        a = tl.load(QM + bh * SQH + m[:, None] * SQM + d[None, :],
                    mask=m[:, None] < M, other=0)
        raw = tl.load(K + b * SKB + h * SKH + n[None, :] * SKN + d[:, None] * SKD,
                      mask=n[None, :] < N, other=0)
        mean = tl.load(KM + bh * SMH + d)
        centered = tl.where(
            n[None, :] < N,
            (raw.to(tl.float32) - mean[:, None].to(tl.float32)).to(tl.bfloat16),
            tl.full((BK, BN), 0, tl.bfloat16))
        acc = tl.dot(a, centered, acc)
    tl.store(O + bh * SOH + m[:, None] * SOM + n[None, :], acc,
             mask=m[:, None] < M)


def _e130_raw_k_delta(qm: torch.Tensor, k: torch.Tensor,
                       km: torch.Tensor) -> torch.Tensor:
    """BF16 correction without materializing a centered/padded BF16 K."""
    b, h, n, d = k.shape
    p = triton.cdiv(n, 128) * 128
    g = p // 128
    if (d != 128 or qm.shape != (b, h, g, d) or
            km.shape != (b, h, 1, d) or not qm.is_contiguous() or
            not km.is_contiguous() or k.dtype != torch.bfloat16):
        raise ValueError("unexpected H3 raw-K correction operands")
    out = torch.empty((b, h, g, p), device=k.device, dtype=torch.bfloat16)
    qflat = qm.view(b * h, g, d)
    kmflat = km.view(b * h, d)
    oflat = out.view(b * h, g, p)
    bm, bn, bk = 32, 128, 32
    _e130_raw_k_delta_kernel[(triton.cdiv(g, bm), triton.cdiv(p, bn), b * h)](
        qflat, k, kmflat, oflat, g, n, h, *k.stride(),
        qflat.stride(0), qflat.stride(1), kmflat.stride(0),
        oflat.stride(0), oflat.stride(1), bm, bn, bk,
        num_warps=4, num_stages=3)
    return out


def _e130_fast_preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              fast_delta_gemm: bool = False,
                              fast_qpack: bool = False,
                              fast_raw_k_delta: bool = False):
    """Exact-output H3 path: fuse Q mean/pad and K center/pad, skip V pad copy.

    This preserves the public path's padded BF16 delta GEMM reduction order.
    Unlike the public path it does not mutate the caller's K tensor.
    """
    b, h, n, d = q.shape
    p = triton.cdiv(n, 128) * 128
    qm = torch.empty((b, h, p // 128, d), device=q.device, dtype=q.dtype)
    if fast_qpack:
        _e130_group_mean_q_only[(b, h, p // 128)](
            q, qm, n, d, *q.stride(), *qm.stride(), num_warps=4)
    else:
        qp = torch.empty((b, h, p, d), device=q.device, dtype=q.dtype)
        _e130_group_mean_pad_q[(b, h, p // 128)](
            q, qp, qm, n, d, *q.stride(), *qp.stride(), *qm.stride(), num_warps=4)

    km = k.mean(dim=2, keepdim=True)
    use_raw_k = fast_raw_k_delta and p >= 2048
    if use_raw_k:
        delta_s = _e130_raw_k_delta(qm, k, km)
    else:
        kp = torch.empty((b, h, p, d), device=k.device, dtype=k.dtype)
        _e130_center_pad_k[(b, h, p // 128)](
            k, km, kp, n, d, *k.stride(), km.stride(0), km.stride(1), km.stride(3),
            *kp.stride(), num_warps=4)
        # Small corrections do not benefit from this launch, and torch's BF16
        # GEMM can use a different reduction path at the shortest boundary.
        delta_s = (_e130_tiled_delta(qm, kp) if fast_delta_gemm and p >= 2048
                   else torch.matmul(qm, kp.transpose(-2, -1)))
    if getattr(fp4attn_cuda, "delta_s_dtype", "fp32") == "fp32":
        delta_s = delta_s.to(torch.float32)
    delta_s = delta_s.contiguous()
    if getattr(fp4attn_cuda, "delta_tail_mask_in_delta", False) and p > n:
        delta_s[..., n:] = float("-inf")

    if fast_qpack:
        qcode = torch.empty((b, h, p, d // 2), device=q.device, dtype=torch.uint8)
        qscale = torch.empty((b, h, p, d // 16), device=q.device, dtype=torch.float8_e4m3fn)
        fp4quant_cuda.scaled_fp4_quant_group_center(q, qm, qcode, qscale)
        qpack = (qcode, qscale)
    else:
        qpack = scale_and_quant_fp4(qp)
    if use_raw_k:
        kcode = torch.empty((b, h, p, d // 2), device=k.device, dtype=torch.uint8)
        kscale = torch.empty((b, h, p, d // 16), device=k.device,
                             dtype=torch.float8_e4m3fn)
        fp4quant_cuda.scaled_fp4_quant_permute_center_readonly(
            k, km, kcode, kscale)
        kpack = (kcode, kscale)
    else:
        kpack = scale_and_quant_fp4_permute(kp)
    vcode = torch.empty((b, h, d, p // 2), device=v.device, dtype=torch.uint8)
    vscale = torch.empty((b, h, d, p // 16), device=v.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(v, vcode, vscale, 1)
    return qpack, kpack, (vcode, vscale), delta_s


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True):
    unpadded_k = k.size(2)

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()
    
    k -= k.mean(dim=-2, keepdim=True)  
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1))
    if getattr(fp4attn_cuda, "delta_s_dtype", "fp32") == "fp32":
        delta_s = delta_s.to(torch.float32)
    delta_s = delta_s.contiguous()
    if getattr(fp4attn_cuda, "delta_tail_mask_in_delta", False) and unpadded_k < delta_s.size(-1):
        delta_s[..., unpadded_k:] = float("-inf")
    return q, k, v, delta_s

def scale_and_quant_fp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def scale_and_quant_fp4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale

def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True
                        ):
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False,
                        per_block_mean = True, fast_preprocess = False,
                        fast_contiguous_output = True, fast_delta_gemm = False,
                        fast_qpack = False, fast_raw_k_delta = False,
                        **kwargs):
    """Sage3 attention, with an experimental exact-output H3 preprocessing opt-in.

    ``fast_preprocess=True`` avoids whole-tensor padding copies. It requires
    noncausal BF16 Q/K/V with matching shapes, head dim 128, and block mean.
    Unlike the default preprocessing path, it does not modify the caller's K.
    ``fast_contiguous_output=False`` leaves the cropped HND result as a view;
    use it only when the caller immediately assembles an output in another layout.
    ``fast_delta_gemm=True`` opt-in replaces the correction matmul with a
    tiled BF16 GEMM; validate on the target shape before deployment.
    ``fast_qpack=True`` computes Q group means first, then fuses BF16
    centering with FP4 packing; it requires a rebuilt fp4quant_cuda extension.
    ``fast_raw_k_delta=True`` (with both preceding fast options) centers K
    during the correction GEMM and FP4 pack without materializing padded K.
    It uses the old path below 2048 padded tokens to preserve short-input
    numerical behavior and requires the rebuilt read-only K pack extension.
    """
    if not fast_preprocess and not fast_contiguous_output:
        raise ValueError("fast_contiguous_output=False requires fast_preprocess=True")
    if fast_delta_gemm and not fast_preprocess:
        raise ValueError("fast_delta_gemm=True requires fast_preprocess=True")
    if fast_qpack and not fast_preprocess:
        raise ValueError("fast_qpack=True requires fast_preprocess=True")
    if fast_qpack and not hasattr(fp4quant_cuda, "scaled_fp4_quant_group_center"):
        raise RuntimeError("fast_qpack=True requires rebuilt fp4quant_cuda extension")
    if fast_raw_k_delta:
        if not (fast_preprocess and fast_delta_gemm and fast_qpack):
            raise ValueError("fast_raw_k_delta=True requires fast_preprocess, fast_delta_gemm and fast_qpack")
        if getattr(fp4attn_cuda, "delta_s_dtype", None) != "bf16":
            raise RuntimeError("fast_raw_k_delta=True requires BF16 delta_s extension")
        if not hasattr(fp4quant_cuda, "scaled_fp4_quant_permute_center_readonly"):
            raise RuntimeError("fast_raw_k_delta=True requires rebuilt read-only K pack extension")
    if fast_preprocess:
        if (is_causal or not per_block_mean or q.ndim != 4
                or q.dtype != torch.bfloat16 or q.size(-1) != 128):
            raise ValueError("fast_preprocess supports noncausal BF16 HND head-dim 128 with per_block_mean")
        if (q.shape != k.shape or q.shape != v.shape
                or k.dtype != q.dtype or v.dtype != q.dtype
                or q.device != k.device or q.device != v.device):
            raise ValueError("fast_preprocess requires matching Q/K/V shape, dtype and device")
    if q.size(-1) >= 256:
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    if fast_preprocess:
        qlist, klist, vlist, delta_s = _e130_fast_preprocess_qkv(
            q, k, v, fast_delta_gemm=fast_delta_gemm, fast_qpack=fast_qpack,
            fast_raw_k_delta=fast_raw_k_delta)
        out = blockscaled_fp4_attn(qlist, klist, vlist, delta_s, KL,
                                  is_causal=False, per_block_mean=True,
                                  is_bf16=True)[0][:, :, :QL, :]
        return out.contiguous() if fast_contiguous_output else out
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
    qlist_from_cuda,
    klist_from_cuda, 
    vlist_from_cuda,
    delta_s,
    KL,
    is_causal,
    per_block_mean,
    is_bf16
    )[0][:, :, :QL, :].contiguous()
    return o_fp4
