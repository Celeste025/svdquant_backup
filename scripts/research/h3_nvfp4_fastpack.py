"""Three-kernel encoder for the frozen H3 legacy NVFP4 activation recipe.

This is an execution helper, not a quantizer change. It matches E005
quantize_pack()['legacy'], including E4M3 RNE, E2M1 ties towards the smaller
signed value, and signed-zero nibbles for negative values rounding to zero.
Input is already smoothed BF16. No activation QDQ tensor is materialized.

The low-level API returns device flags without synchronizing. The adapter
checks them immediately unless collect_fastpack_checks() defers validation
until its exit. Nonfinite input and nonzero groups whose E4M3 scale underflows
to zero are outside the validated domain and MUST stop the caller.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

import torch
import triton
import triton.language as tl


class FastPacked(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    domain_flags: torch.Tensor


@triton.jit
def _amax_parts(X, Partial, N: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + ix, ix < N, 0).to(tl.float32)
    tl.store(Partial + tl.program_id(0), tl.max(tl.abs(x), 0))


@triton.jit
def _global_scale(Partial, Global, Flags, P: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.arange(0, BLOCK)
    mx = tl.max(tl.load(Partial + ix, ix < P, 0), 0)
    # PyTorch CUDA division by a Python scalar is FP32 reciprocal multiply.
    # H3 has ONE division by 2688, unlike Wan's successive /6 and /448.
    g = tl.maximum(tl.maximum(mx, 1.0e-12) * (1.0 / 2688.0), 1.0e-12)
    invalid = (~((mx >= 0) & (mx < float('inf')))) | (~((g > 0) & (g < float('inf'))))
    tl.store(Global, g)
    tl.store(Flags, invalid.to(tl.int32))
    tl.store(Flags + 1, 0)


@triton.jit
def _encode_groups(X, Global, Codes, Scales, Flags, M: tl.constexpr, K: tl.constexpr,
                   CP: tl.constexpr, RP: tl.constexpr, GROUPS: tl.constexpr):
    idx = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    r, c = idx // CP, idx % CP
    lane = tl.arange(0, 16)
    valid = (r < M) & (c < K // 16)
    x = tl.load(X + r[:, None] * K + c[:, None] * 16 + lane[None, :],
                valid[:, None], 0).to(tl.float32)
    invalid_input = valid[:, None] & ((x != x) | (tl.abs(x) == float('inf')))
    if tl.sum(tl.sum(invalid_input.to(tl.int32), 1), 0) > 0:
        tl.atomic_or(Flags, 1)
    amax = tl.max(tl.abs(x), 1)
    g = tl.load(Global)
    ideal = tl.maximum(amax * (1.0 / 6.0), 1.0e-12)
    u = tl.maximum(tl.div_rn(ideal, g), 1.1754943508222875e-38)
    sf = u.to(tl.float8e4nv).to(tl.float32)
    effective = sf * g
    # E005 gives zero-scale groups a canonical +0 code; reject the nonzero
    # input subset below rather than silently treating it as valid packing.
    z = tl.where(effective[:, None] == 0, 0.0, tl.div_rn(x, effective[:, None]))
    bad_scale = valid & ((sf != sf) | (tl.abs(sf) == float('inf')))
    bad_z = valid[:, None] & ((z != z) | (tl.abs(z) == float('inf')))
    if tl.sum(bad_scale.to(tl.int32), 0) + tl.sum(tl.sum(bad_z.to(tl.int32), 1), 0) > 0:
        tl.atomic_or(Flags, 1)
    unrepresentable = valid & (sf == 0) & (amax != 0)
    if tl.sum(unrepresentable.to(tl.int32), 0) > 0:
        tl.atomic_or(Flags + 1, 1)
    az, neg = tl.abs(z), z < 0
    code = tl.full((GROUPS, 16), 0, tl.int32)
    for mid in tl.static_range(7):
        threshold = 0.25 if mid == 0 else (0.75 if mid == 1 else (1.25 if mid == 2 else (1.75 if mid == 3 else (2.5 if mid == 4 else (3.5 if mid == 5 else 5.0)))))
        code += tl.where(neg, az >= threshold, az > threshold).to(tl.int32)
    # E005 keeps -0 for negative values rounded to zero. Actual +/-0 input
    # has z < 0 false and therefore encodes as +0. Do not canonicalize here.
    code |= tl.where(neg, 8, 0)
    shifted = code << ((lane[None, :] % 2) * 4)
    packed = tl.sum(tl.reshape(shifted, (GROUPS, 8, 2)), 2).to(tl.uint8)
    pair = tl.arange(0, 8)
    tl.store(Codes + r[:, None] * (K // 2) + c[:, None] * 8 + pair[None, :],
             packed, valid[:, None])
    # F.SwizzleType.SWIZZLE_32_4_4, including explicit zero padding.
    offset = (r // 128) * 128 * CP + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    tl.store(Scales + offset, tl.where(valid, sf, 0).to(tl.float8e4nv), r < RP)


def pack_legacy_h3(x: torch.Tensor) -> FastPacked:
    """Pack contiguous CUDA BF16 [..., K], K divisible by 16; no host sync.

    Returns packed uint8 [M,K/2], swizzled E4M3 scales (with zero padding),
    FP32 decode global [1], and int32 flags [invalid/nonfinite, nonzero-SF0].
    Call validate_flags before using a result as validated research evidence.
    """
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('Expected contiguous CUDA BF16 input')
    if x.numel() == 0 or x.ndim < 2:
        raise ValueError('Expected a nonempty matrix or batched matrix')
    k = x.shape[-1]
    m = x.numel() // k
    if k % 16:
        raise ValueError('K must be divisible by 16')
    cp, rp = triton.cdiv(k // 16, 4) * 4, triton.cdiv(m, 128) * 128
    count = triton.cdiv(x.numel(), 16384)
    parts = torch.empty((count,), device=x.device, dtype=torch.float32)
    global_scale = torch.empty((1,), device=x.device, dtype=torch.float32)
    flags = torch.empty((2,), device=x.device, dtype=torch.int32)
    codes = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((rp * cp,), device=x.device, dtype=torch.float8_e4m3fn)
    _amax_parts[(count,)](x, parts, x.numel(), 16384, num_warps=8, enable_fp_fusion=False)
    _global_scale[(1,)](parts, global_scale, flags, count, triton.next_power_of_2(count), enable_fp_fusion=False)
    _encode_groups[(triton.cdiv(rp * cp, 128),)](x, global_scale, codes, scales, flags, m, k, cp, rp, 128,
                                              num_warps=4, enable_fp_fusion=False)
    return FastPacked(codes, scales, global_scale, flags)


_CHECKS = ContextVar('h3_nvfp4_checks', default=None)


def validate_flags(flags):
    """Synchronizing domain check; never discard a failed flag."""
    if flags and bool(torch.stack(flags).ne(0).any()):
        raise ValueError('H3 legacy NVFP4 invalid/nonfinite input or nonzero group with zero E4M3 scale')


@contextmanager
def collect_fastpack_checks():
    """Validate at exit; place any timing-end synchronization INSIDE scope."""
    checks = []
    token = _CHECKS.set(checks)
    try:
        yield checks
        validate_flags(checks)
    finally:
        _CHECKS.reset(token)


def pack_activation_fast(x: torch.Tensor, *, chunk_rows=128):
    """NativeH3Linear adapter; chunk_rows is accepted for interface parity."""
    from wan_native_nvfp4 import PackedNVFP4
    if x.ndim < 2 or x.shape[-1] % 32 or chunk_rows < 1:
        raise ValueError('Native H3 requires K divisible by 32 and positive chunk_rows')
    p = pack_legacy_h3(x)
    checks = _CHECKS.get()
    if checks is None:
        validate_flags([p.domain_flags])
    else:
        checks.append(p.domain_flags)
    return PackedNVFP4(p.codes, None, p.global_scale, p.scales,
                       tuple(x.shape), 'h3_legacy_e005_triton')
