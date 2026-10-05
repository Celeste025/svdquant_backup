"""E080 diagnostic helpers; no FlashInfer imports or implicit CUDA use.

P follows Sage3's *unrounded* scale normalization; Q/K/V use the rounded
E4M3 scale. P must already contain the online-softmax 2688-scaled exp values.
These helpers do not reproduce native QK accumulation/exp2 or claim end-to-end
byte parity. Return values are logical FP32 tensors, not optimized kernels.
"""
from __future__ import annotations

import torch


def logical_scales(buffer):
    """Undo official Sage3's 64-row/4-column scale-byte swizzle."""
    *leading, rows, cols = buffer.shape
    if rows % 64 or cols % 4:
        raise ValueError(f"Unsupported swizzled scale shape: {tuple(buffer.shape)}")
    r = torch.arange(rows, device=buffer.device)[:, None]
    c = torch.arange(cols, device=buffer.device)[None, :]
    offset = (r // 64) * 64 * cols + (c // 4) * 256
    offset = offset + (r % 16) * 16 + ((r % 64) // 16) * 4 + c % 4
    raw = buffer.contiguous().view(torch.uint8).reshape(-1, rows * cols)
    return raw[:, offset.flatten()].view(torch.float8_e4m3fn).reshape(
        *leading, rows, cols)


def k_permutation(length, device=None):
    """Physical K row -> original logical row within each 32-key block."""
    if length % 32:
        raise ValueError("K length must be padded to a multiple of 32")
    t = torch.arange(length, device=device, dtype=torch.long)
    u = t % 32
    return (t // 32) * 32 + (u // 8) * 2 + ((u % 8) // 2) * 8 + u % 2


def decode_e2m1(codes):
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                           -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                          device=codes.device, dtype=torch.float32)
    return levels[codes.long()]


def decode_packed(packed, scales, kind="q"):
    """Decode official packed Q/K/V to logical [B,H,N,D] FP32.

    V input packs/scales have transposed shape [B,H,D,N/2 or N/16].
    K's physical row permutation is undone after decoding both codes/scales.
    """
    kind = kind.lower()
    if kind not in {"q", "k", "v"} or packed.dtype != torch.uint8:
        raise ValueError("Expected uint8 packed data and kind q, k, or v")
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    sf = logical_scales(scales).float()
    if codes.shape[:-1] != sf.shape[:-1] or codes.shape[-1] != sf.shape[-1] * 16:
        raise ValueError("Packed values/scales have inconsistent shapes")
    out = decode_e2m1(codes) * sf.repeat_interleave(16, -1)
    if kind == "k":
        inverse = k_permutation(out.shape[-2], out.device).argsort()
        out = out.index_select(-2, inverse)
    elif kind == "v":
        out = out.transpose(-2, -1)
    return out.contiguous()


def e2m1_codes(values):
    """Finite signed FP32 -> E2M1 nibble, RN ties-even, saturation at +/-6."""
    mids = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.],
                        device=values.device, dtype=torch.float32)
    magnitude = values.float().abs().contiguous()
    lower = torch.bucketize(magnitude, mids, right=False)
    midpoint = (lower < 7) & (magnitude == mids[lower.clamp_max(6)])
    code = lower + (midpoint & ((lower % 2) == 1)).long()
    return code.to(torch.uint8) | (torch.signbit(values).to(torch.uint8) << 3)


def _quant_dequant(x, *, rounded_normalizer, return_details):
    if x.shape[-1] % 16:
        raise ValueError("The quantized dimension must be a multiple of 16")
    values = x.float().reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    raw_scales = values.abs().amax(-1) / 6.
    # CUDA's conversion saturates; ordinary torch .to(float8) alone can NaN.
    scales = raw_scales.clamp(0., 448.).to(torch.float8_e4m3fn)
    if rounded_normalizer:
        sf = scales.float()
        reciprocal = torch.where(sf == 0., torch.zeros_like(sf), sf.reciprocal())
        normalized = values * reciprocal.unsqueeze(-1)
    else:
        # P divides by the FP32 scale before E4M3 rounding. For a zero group,
        # choose signed zero codes explicitly through division by one.
        denominator = torch.where(raw_scales == 0., torch.ones_like(raw_scales), raw_scales)
        normalized = values / denominator.unsqueeze(-1)
    codes = e2m1_codes(normalized)
    result = (decode_e2m1(codes) * scales.float().unsqueeze(-1)).reshape(x.shape)
    if return_details:
        return result, {"codes": codes.reshape(x.shape), "scales": scales,
                        "raw_scales": raw_scales}
    return result


def p_quant_dequant(x, *, return_details=False):
    """Quantize already-2688-scaled exp or signed P coefficients along -1."""
    return _quant_dequant(x.float(), rounded_normalizer=False,
                         return_details=return_details)


def v_quant_dequant(x, *, return_details=False):
    """Quantize along -1 AFTER BF16 input rounding, as the official V pack.

    Pass [B,H,D,N] (transpose logical V yourself); groups run along keys.
    """
    return _quant_dequant(x.to(torch.bfloat16).float(), rounded_normalizer=True,
                         return_details=return_details)


def pair_forward(x, key_dim=-1, side="p"):
    """Each 32-key block: adjacent pairs -> 16 sums then 16 differences.

    P uses half-sum/half-difference; V uses full sum/difference. Their dot
    product is invariant in exact arithmetic. Arithmetic here is FP32.
    """
    if side.lower() not in {"p", "v"}:
        raise ValueError("side must be p or v")
    y = x.float().movedim(key_dim, -1)
    if y.shape[-1] % 32:
        raise ValueError("Pair encoding requires a multiple of 32 keys")
    grouped = y.reshape(*y.shape[:-1], y.shape[-1] // 32, 16, 2)
    a, b = grouped[..., 0], grouped[..., 1]
    encoded = torch.cat((a + b, a - b), -1)
    if side.lower() == "p":
        encoded = encoded * .5
    return encoded.reshape(y.shape).movedim(-1, key_dim).contiguous()


def inverse_P(x, key_dim=-1):
    """Restore logical P order from half-sum/half-difference coefficients."""
    y = x.float().movedim(key_dim, -1)
    if y.shape[-1] % 32:
        raise ValueError("Pair decoding requires a multiple of 32 keys")
    grouped = y.reshape(*y.shape[:-1], y.shape[-1] // 32, 32)
    plus, minus = grouped[..., :16], grouped[..., 16:]
    decoded = torch.stack((plus + minus, plus - minus), -1)
    return decoded.reshape(y.shape).movedim(-1, key_dim).contiguous()


def within32_permutation(length, device=None, seed=8007):
    """Fixed independent permutations per 32-key block; no global RNG change."""
    if length % 32:
        raise ValueError("Permutation requires a multiple of 32 keys")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    local = torch.rand((length // 32, 32), generator=generator).argsort(-1)
    indices = (local + torch.arange(length // 32)[:, None] * 32).flatten()
    return indices.to(device=device)


def permute_within32(x, key_dim=-1, inverse=False, seed=8007):
    indices = within32_permutation(x.shape[key_dim], x.device, seed)
    return x.index_select(key_dim, indices.argsort() if inverse else indices)
