"""E009 explicit H3 zero-SF compatibility variant; frozen sources unchanged.

This applies ONLY to the historical H3 common.py NVFP4 recipe. There is no
DeepCompressor/Wan remove_zero effective-scale replacement here. At E4M3 SF=0,
finite H3 inputs yield inf/NaN normalized values; the historical signed15
argmin selects index0 and reconstruction is -6*0 == -0. E005 encodes a
canonical +0 nibble. These are numerically equal, not zero-sign bitwise equal.

The independently frozen E005 encoder is the slow reference. The fast adapter
calls the frozen H3 low-level helper, still rejects flag[0] (invalid/nonfinite),
and explicitly permits flag[1] (nonzero-input groups with zero E4M3 SF).
It counts affected CALLS, not affected groups; the old helper exposes one bit
per call. Slow E005 diagnostic statistics expose exact affected-group counts.

Do not use this variant for a new full-model run until a saved real failing
input passes independent old-QDQ/decode finite+numeric-equality gates. Preserve
the strict-domain failed run. This module does not change weights or globals.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from wan_native_nvfp4 import PackedNVFP4, swizzle_scales

COMPATIBILITY = 'h3_e005_sf0_canonical_zero_numeric_equivalence_v1'
_CHECKS = ContextVar('h3_zero_sf_compat_checks', default=None)


class ZeroSFCheckLog(list):
    """One original int32[2] flags tensor per fast pack, plus exit summary."""
    def __init__(self):
        super().__init__()
        self.summary = None

    def validate(self):
        if self:
            if any(f.dtype != torch.int32 or f.numel() != 2 for f in self):
                raise ValueError('Unexpected frozen H3 fast-helper flag ABI')
            # A single synchronization/CPU copy at scope exit; include its cost.
            flags = torch.stack([f.reshape(2) for f in self]).cpu()
            if bool(flags[:, 0].ne(0).any()):
                raise ValueError('H3 zero-SF compatibility still rejects invalid/nonfinite input or scales')
            if bool(((flags[:, 1] != 0) & (flags[:, 1] != 1)).any()):
                raise ValueError('Unknown zero-SF flag bit pattern')
            affected = flags[:, 1].nonzero().flatten().tolist()
        else:
            affected = []
        self.summary = {
            'compatibility': COMPATIBILITY, 'checked_calls': len(self),
            'invalid_calls': 0, 'calls_with_nonzero_input_zero_sf': len(affected),
            'affected_call_indices_zero_based': affected,
            'count_scope': 'Call-level flags only; not a count of affected tensor groups or elements',
        }
        return self.summary


@contextmanager
def collect_fastpack_checks():
    """Collect 200 flags and validate on exit; end timing/sync inside scope.

    The yielded list has `.summary` after context exit, retaining affected-call
    indices so this expanded domain is never silently treated as flag-free.
    """
    log = ZeroSFCheckLog()
    token = _CHECKS.set(log)
    try:
        yield log
        log.validate()
    finally:
        _CHECKS.reset(token)


@torch.inference_mode()
def pack_activation_legacy_zero_sf(x: torch.Tensor, *, chunk_rows: int = 128) -> PackedNVFP4:
    """Use E005 itself, rather than another implementation of its rounding."""
    from probe_h3_native_contract import quantize_pack
    if x.dtype != torch.bfloat16 or x.ndim < 2 or x.shape[-1] % 32 or x.numel() == 0:
        raise ValueError('Expected nonempty BF16 H3 [..., K], K divisible by32')
    if chunk_rows < 1 or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite input and positive chunk_rows required')
    packed = quantize_pack(x.reshape(-1, x.shape[-1]), chunk=chunk_rows)
    sf, global_scale = packed['scale'], packed['global']
    if not bool(torch.isfinite(sf.float()).all()) or not bool(torch.isfinite(global_scale).all()) or not bool((global_scale > 0).all()):
        raise ValueError('Nonfinite scale/global from E005 reference')
    if packed['stats']['nonfinite_normalized_values']:
        raise ValueError('E005 canonicalized normalized values remain nonfinite')
    result = PackedNVFP4(packed['legacy'], None, global_scale, swizzle_scales(sf),
                         tuple(x.shape), COMPATIBILITY)
    result.zero_sf_statistics = dict(packed['stats'])
    return result


@torch.inference_mode()
def pack_activation_fast_zero_sf(x: torch.Tensor, *, chunk_rows: int = 128) -> PackedNVFP4:
    """Frozen low-level codes/SF/global; explicit new validation semantics."""
    from h3_nvfp4_fastpack import pack_legacy_h3
    del chunk_rows
    if not x.is_contiguous():
        x = x.contiguous()
    packed = pack_legacy_h3(x)
    log = _CHECKS.get()
    standalone = log is None
    if standalone:
        log = ZeroSFCheckLog()
    log.append(packed.domain_flags)
    result = PackedNVFP4(packed.codes, None, packed.global_scale, packed.scales,
                         tuple(x.shape), COMPATIBILITY)
    if standalone:
        result.zero_sf_statistics = log.validate()
    return result


# Explicit short aliases for injection into frozen NativeH3Linear.
pack_activation_legacy = pack_activation_legacy_zero_sf
pack_activation_fast = pack_activation_fast_zero_sf
