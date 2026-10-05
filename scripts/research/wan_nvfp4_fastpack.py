"""Minimal native NVFP4 activation packer for the legacy Wan recipe.

Research baseline, not a new quantizer. Three GPU kernels implement global amax,
FP32 global scale, and group-16 E4M3/E2M1 packing. Midpoint ties go to the larger
signed value, matching the vendored DeepCompressor CUDA codebook kernel. No
smoothing, clipping search, host synchronization, or dequantized activation.
All-zero inputs use an equivalent canonical zero representation. Device flags
reject unrepresentable zero-SF/nonzero-code and invalid global cases. Checks
can be collected and read once after a forward, outside a timing interval.
"""
from __future__ import annotations
from typing import NamedTuple
from contextlib import contextmanager
from contextvars import ContextVar
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
    # CUDA Tensor / Python scalar in PyTorch uses float32 reciprocal multiply.
    # Keep the two steps separate; this is not mx / (6 * 448).
    s = mx * (1.0 / 6.0)
    g = s * (1.0 / 448.0)
    invalid = (~((mx >= 0) & (mx < float('inf')))) | ((mx > 0) & (g == 0))
    tl.store(Flags, invalid.to(tl.int32))
    tl.store(Flags + 1, 0)
    # Canonical zero is numerically equal to the legacy all-zero QDQ result;
    # its global/SF bytes need not equal legacy's degenerate leaves.
    tl.store(Global, tl.where(mx == 0, 1.0, g))


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
    ideal = amax * (1.0 / 6.0)
    u = tl.minimum(tl.div_rn(ideal, g), 448.0)
    # Positive E4M3 quantization, with ties upwards including subnormals.
    bits = u.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127
    step_exp = tl.maximum(exponent - 3, -9)
    step = ((step_exp + 127) << 23).to(tl.float32, bitcast=True)
    units = tl.div_rn(u, step)
    low = tl.floor(units)
    sf = (low + ((units - low) >= 0.5).to(tl.float32)) * step
    # The vendored signed E4M3 codebook represents its sole zero as -0.
    sf = tl.where(sf == 0, tl.full((GROUPS,), -2147483648, tl.int32).to(tl.float32, bitcast=True), sf)
    effective = sf * g
    # Legacy QuantScale.remove_zero changes effective scale, not the leaves.
    effective = tl.where(effective == 0, 1.0, effective)
    z = tl.div_rn(x, effective[:, None])
    az = tl.abs(z)
    neg = z < 0
    # Negative midpoint chooses the smaller magnitude (larger signed value).
    code = tl.full((GROUPS, 16), 0, tl.int32)
    for mid in tl.static_range(7):
        threshold = 0.25 if mid == 0 else (0.75 if mid == 1 else (1.25 if mid == 2 else (1.75 if mid == 3 else (2.5 if mid == 4 else (3.5 if mid == 5 else 5.0)))))
        code += tl.where(neg, az > threshold, az >= threshold).to(tl.int32)
    unrepresentable = valid & (sf == 0) & (tl.max(code, 1) != 0)
    if tl.sum(unrepresentable.to(tl.int32), 0) > 0:
        tl.atomic_or(Flags + 1, 1)
    # Use a canonical +0 nibble; both hardware signs decode to exact zero.
    code |= tl.where(neg & (code != 0), 8, 0)
    shifted = code << ((lane[None, :] % 2) * 4)
    packed = tl.sum(tl.reshape(shifted, (GROUPS, 8, 2)), 2).to(tl.uint8)
    pair = tl.arange(0, 8)
    tl.store(Codes + r[:, None] * (K // 2) + c[:, None] * 8 + pair[None, :],
             packed, valid[:, None])
    # PyTorch F.SwizzleType.SWIZZLE_32_4_4; 128-row/4-column tiles.
    offset = (r // 128) * 128 * CP + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    tl.store(Scales + offset, tl.where(valid, sf, 0).to(tl.float8e4nv), r < RP)


def pack_legacy_wan(x: torch.Tensor) -> FastPacked:
    """Pack contiguous BF16 input, consuming existing smooth x_s.

    Strict recipe compatibility is established by test_wan_nvfp4_fastpack.py;
    this low-level function returns value-domain flags without CPU sync.
    Returned scale storage includes fully initialized zero padding.
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
    partial_count = triton.cdiv(x.numel(), 16384)
    parts = torch.empty((partial_count,), device=x.device, dtype=torch.float32)
    g = torch.empty((1,), device=x.device, dtype=torch.float32)
    flags = torch.empty((2,), device=x.device, dtype=torch.int32)
    codes = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((rp * cp,), device=x.device, dtype=torch.float8_e4m3fn)
    _amax_parts[(partial_count,)](x, parts, x.numel(), 16384, num_warps=8, enable_fp_fusion=False)
    _global_scale[(1,)](parts, g, flags, partial_count, triton.next_power_of_2(partial_count), enable_fp_fusion=False)
    _encode_groups[(triton.cdiv(rp * cp, 128),)](x, g, codes, scales, flags, m, k, cp, rp, 128, num_warps=4, enable_fp_fusion=False)
    return FastPacked(codes, scales, g, flags)


_CHECKS = ContextVar('wan_nvfp4_checks', default=None)


def validate_flags(flags):
    if flags and bool(torch.stack(flags).ne(0).any()):
        raise ValueError('NVFP4 legacy recipe cannot be represented: invalid global or zero SF with nonzero codes')


@contextmanager
def collect_fastpack_checks():
    """Validate all calls at exit; put timing-end synchronization INSIDE scope."""
    checks = []
    token = _CHECKS.set(checks)
    try:
        yield checks
        validate_flags(checks)
    finally:
        _CHECKS.reset(token)


def pack_activation_fast(x: torch.Tensor, *, quantizer=None, chunk_rows=128):
    """Adapter for NativeWanLinear; recipe compatibility is validated separately."""
    from wan_native_nvfp4 import PackedNVFP4
    if x.shape[-1] % 32:
        raise ValueError('Native Wan GEMM requires K divisible by 32')
    packed = pack_legacy_wan(x)
    checks = _CHECKS.get()
    if checks is None:
        validate_flags([packed.domain_flags])
    else:
        checks.append(packed.domain_flags)
    return PackedNVFP4(packed.codes, None, packed.global_scale, packed.scales,
                       tuple(x.shape), 'legacy_wan_triton_verified')


def validate_quantizer_contract(quantizer):
    """One-time offline gate before assigning this callable to a model layer."""
    from deepcompressor.data.dtype import QuantDataType
    from deepcompressor.quantizer.kernel.rtn import QuantRtnKernel
    q, c = quantizer, quantizer.config
    if c is None or c.dtype != QuantDataType.from_str('sfp4_e2m1_all') or c.zero_domain is not None:
        raise ValueError('Expected signed E2M1 without zero point')
    groups = c.group_shapes
    if len(groups) != 2 or tuple(groups[0][:2]) != (-1,-1) or tuple(groups[1][:2]) != (1,16):
        raise ValueError('Expected tensor-global then row/group16 scales')
    if any(v not in (-1,1) for shape in groups for v in shape[2:]):
        raise ValueError('Unsupported nontrivial trailing grouping')
    if tuple(c.scale_dtypes) != (torch.float32, QuantDataType.from_str('sfp8_e4m3_nan')):
        raise ValueError('Expected FP32 global and E4M3 block scales')
    if q.channels_dim != -1 or q.develop_dtype != torch.float32:
        raise ValueError('Expected last-channel FP32 development')
    if len(c.decompose().steps) != 1:
        raise ValueError('Progressive quantization is not supported')
    if q.kernel is not None and type(q.kernel) is not QuantRtnKernel:
        raise ValueError('Custom quantization kernel is not supported')
    for name in ('scale','zero','dynamic_range','range_bound','quant_range'):
        if getattr(q,name,None) is not None:
            raise ValueError(f'Unsupported explicit {name}: fast path only implements the frozen dynamic recipe')
    return True
