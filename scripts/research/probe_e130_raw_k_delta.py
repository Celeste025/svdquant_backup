#!/usr/bin/env python3
"""Test a BF16 correction GEMM that centers raw K during its tile load."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = {
    "e098": Path("/data1/models/svdquant-wjq/research/20261014/E098/"
                 "h3_p1_s0_b25_480x832_73f_qkv.pt"),
    "e121": Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/"
                 "race/block25_qkv_h56.pt"),
}


@triton.jit
def _raw_k_delta(QM, K, KM, O, M: tl.constexpr, N: tl.constexpr,
                 H: tl.constexpr, SKB: tl.constexpr, SKH: tl.constexpr,
                 SKN: tl.constexpr, SKD: tl.constexpr,
                 SQH: tl.constexpr, SQM: tl.constexpr,
                 SMH: tl.constexpr, SOH: tl.constexpr, SOM: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
        centered = tl.where(n[None, :] < N,
                            (raw.to(tl.float32) - mean[:, None].to(tl.float32)).to(tl.bfloat16),
                            tl.full((BK, BN), 0, tl.bfloat16))
        acc = tl.dot(a, centered, acc)
    tl.store(O + bh * SOH + m[:, None] * SOM + n[None, :], acc,
             mask=m[:, None] < M)


def time_cuda(fn, warmup: int = 10, batches: int = 8, repeats: int = 12) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(batches):
        events = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        means.append(statistics.mean(start.elapsed_time(end) for start, end in events))
    return {"mean_ms": statistics.mean(means),
            "cv_batch_percent": 100 * statistics.stdev(means) / statistics.mean(means),
            "batch_means_ms": means, "calls": batches * repeats}


def raw_k_delta(qm: torch.Tensor, k: torch.Tensor,
                km: torch.Tensor) -> torch.Tensor:
    """Produce padded BF16 delta_s without a centered/padded BF16 K tensor."""
    b, h, n, d = k.shape
    p = triton.cdiv(n, 128) * 128
    g = p // 128
    if (d != 128 or qm.shape != (b, h, g, d) or
            km.shape != (b, h, 1, d) or not qm.is_contiguous() or
            not km.is_contiguous()):
        raise ValueError("unexpected H3 raw-K correction operands")
    out = torch.empty((b, h, g, p), device=k.device, dtype=torch.bfloat16)
    qflat = qm.view(b * h, g, d)
    kmflat = km.view(b * h, d)
    oflat = out.view(b * h, g, p)
    bm, bn, bk = 32, 128, 32
    _raw_k_delta[(triton.cdiv(g, bm), triton.cdiv(p, bn), b * h)](
        qflat, k, kmflat, oflat, g, n, h, *k.stride(),
        qflat.stride(0), qflat.stride(1), kmflat.stride(0),
        oflat.stride(0), oflat.stride(1), bm, bn, bk,
        num_warps=4, num_stages=3)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=CAPTURES, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--sage3-dir", type=Path, default=ROOT / "third_party/sage3_h3_maskfix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    from sageattn3.api import (_e130_center_pad_k, _e130_group_mean_pad_q,
                               _e130_tiled_delta)

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURES[args.capture], map_location="cpu", weights_only=True,
                         mmap=True)
    if payload["q"].ndim == 3:
        q, k = (payload[name][:args.tokens].to("cuda:0").transpose(0, 1).unsqueeze(0)
                for name in ("q", "k"))
    else:
        q, k = (payload[name][:, :, :args.tokens].to("cuda:0")
                for name in ("q", "k"))
    b, h, n, d = q.shape
    if n != args.tokens or q.shape != k.shape or d != 128:
        raise RuntimeError("unexpected capture shape")
    p = triton.cdiv(n, 128) * 128
    g = p // 128
    qp = torch.empty((b, h, p, d), device=q.device, dtype=q.dtype)
    qm = torch.empty((b, h, g, d), device=q.device, dtype=q.dtype)
    _e130_group_mean_pad_q[(b, h, g)](
        q, qp, qm, n, d, *q.stride(), *qp.stride(), *qm.stride(), num_warps=4)
    km = k.mean(dim=2, keepdim=True)
    kp = torch.empty((b, h, p, d), device=q.device, dtype=q.dtype)
    out = torch.empty((b, h, g, p), device=q.device, dtype=torch.bfloat16)
    qflat = qm.view(b * h, g, d)
    kmflat = km.view(b * h, d)
    oflat = out.view(b * h, g, p)
    bm, bn, bk = 32, 128, 32

    def center_pad() -> None:
        _e130_center_pad_k[(b, h, g)](
            k, km, kp, n, d, *k.stride(), km.stride(0), km.stride(1), km.stride(3),
            *kp.stride(), num_warps=4)

    def padded_gemm() -> torch.Tensor:
        return _e130_tiled_delta(qm, kp)

    def raw_gemm() -> None:
        _raw_k_delta[(triton.cdiv(g, bm), triton.cdiv(p, bn), b * h)](
            qflat, k, kmflat, oflat, g, n, h, *k.stride(),
            qflat.stride(0), qflat.stride(1), kmflat.stride(0),
            oflat.stride(0), oflat.stride(1), bm, bn, bk,
            num_warps=4, num_stages=3)

    with torch.inference_mode():
        center_pad()
        reference = padded_gemm()
        raw_gemm()
        torch.cuda.synchronize()
        difference = out.float() - reference.float()
        mismatch = out != reference
        result = {
            "format": "e130-raw-k-tiled-delta-probe-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "capture": str(CAPTURES[args.capture]), "sage3_dir": str(args.sage3_dir.resolve()),
            "shape_hnd": list(q.shape), "padded_tokens": p,
            "bitwise_equal_vs_padded_tiled": bool(torch.equal(out, reference)),
            "mismatch_fraction": float(mismatch.float().mean()),
            "nmse_vs_padded_tiled": float(difference.square().sum() / reference.float().square().sum()),
            "max_abs_difference": float(difference.abs().max()),
            "timings": {},
        }
        # Compare just the reusable components. Allocation and K quant-pack
        # are excluded here; API-level end-to-end timing requires a later gate.
        for order in (("padded", "raw"), ("raw", "padded")):
            for arm in order:
                fn = (lambda: (center_pad(), padded_gemm())) if arm == "padded" else raw_gemm
                result["timings"].setdefault(arm, []).append(time_cuda(fn))
        result["mean_ms"] = {arm: statistics.mean(row["mean_ms"] for row in entries)
                             for arm, entries in result["timings"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("shape_hnd", "bitwise_equal_vs_padded_tiled",
                       "mismatch_fraction", "nmse_vs_padded_tiled", "mean_ms")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
