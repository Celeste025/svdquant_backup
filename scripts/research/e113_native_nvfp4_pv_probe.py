#!/usr/bin/env python3
"""Minimal Blackwell NVFP4 (E2M1 + E4M3/16) P@V native-MMA probe."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _nvfp4_pv_tl(A, B, SA, SB, O):
    m = tl.arange(0, 128)
    kb = tl.arange(0, 64)
    n = tl.arange(0, 128)
    kg = tl.arange(0, 8)
    a = tl.load(A + m[:, None] * 64 + kb[None, :])
    b = tl.load(B + kb[:, None] * 128 + n[None, :])
    sa = tl.load(SA + m[:, None] * 8 + kg[None, :])
    sb = tl.load(SB + n[:, None] * 8 + kg[None, :])
    c = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1")
    tl.store(O + m[:, None] * 128 + n[None, :], c)


def pack_e2m1(x: torch.Tensor):
    ax = x.abs()
    code = torch.zeros_like(ax, dtype=torch.uint8)
    for cutoff, value in ((.25, 1), (.75, 2), (1.25, 3), (1.75, 4),
                          (2.5, 5), (3.5, 6), (5.0, 7)):
        code = torch.where(ax >= cutoff if value in (1, 3, 5, 7) else ax > cutoff,
                           value, code)
    code |= (x < 0).to(torch.uint8) * 8
    return (code[:, ::2] | (code[:, 1::2] << 4)).contiguous()


def main():
    torch.manual_seed(37)
    a = torch.randn((128, 128), device="cuda", dtype=torch.float32).abs()
    b = torch.randn((128, 128), device="cuda", dtype=torch.float32)
    a_s = (a.reshape(128, 8, 16).amax(-1) / 6).to(torch.float8_e4m3fn)
    b_s = (b.reshape(128, 8, 16).abs().amax(-1) / 6).to(torch.float8_e4m3fn)
    a_s_f = a_s.float().repeat_interleave(16, -1)
    b_s_f = b_s.float().repeat_interleave(16, -1)
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/MixAttention"))
    from pv_hadamard.quant import round_rne_e2m1
    a_code = round_rne_e2m1(a / a_s_f)
    b_code = round_rne_e2m1(b / b_s_f)
    ap = pack_e2m1(a_code)
    bp = pack_e2m1(b_code).T.contiguous()
    out = torch.empty((128, 128), device="cuda", dtype=torch.float32)
    kernel = _nvfp4_pv_tl[(1,)](ap, bp, a_s, b_s, out)
    ref = (a_code * a_s_f) @ (b_code * b_s_f).T
    torch.cuda.synchronize()
    nmse = ((out - ref).square().sum() / ref.square().sum()).item()
    print({"gpu": torch.cuda.get_device_name(), "nmse": nmse,
           "max_abs": (out - ref).abs().max().item(),
           "finite": bool(torch.isfinite(out).all().item()),
           "native_mma": "mma.sync" in kernel.asm["ptx"] or "tcgen05.mma" in kernel.asm["ptx"],
           "nvfp4_block16": "scale_vec::4X" in kernel.asm["ptx"]})
    print([line.strip() for line in kernel.asm["ptx"].splitlines()
           if "mma." in line or "scale_vec" in line][:8])
    assert ("mma.sync" in kernel.asm["ptx"] or "tcgen05.mma" in kernel.asm["ptx"])
    assert "scale_vec::4X" in kernel.asm["ptx"]
    assert torch.isfinite(out).all() and nmse < 1e-5


if __name__ == "__main__":
    main()
