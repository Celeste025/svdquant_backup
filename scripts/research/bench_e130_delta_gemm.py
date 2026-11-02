#!/usr/bin/env python3
"""Probe bitwise correctness and speed of tiled BF16 Sage3 delta_s GEMM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261023/"
               "E121_h3_split_generation/race/block25_qkv_h56.pt")
SAGE3 = ROOT / "third_party/sage3_h3_maskfix"


@triton.jit
def _delta_gemm(QM, K, O, M: tl.constexpr, N: tl.constexpr,
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


def measure(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(8):
        pairs = []
        for _ in range(12):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        means.append(statistics.mean(start.elapsed_time(end) for start, end in pairs))
    return {"mean_ms": statistics.mean(means),
            "cv_batch_percent": 100 * statistics.stdev(means) / statistics.mean(means),
            "batch_means_ms": means}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage3-dir", type=Path, default=SAGE3)
    parser.add_argument("--capture", type=Path, default=CAPTURE)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    sys.path.insert(0, str(opts.sage3_dir))
    from sageattn3.api import _e130_group_mean_pad_q, _e130_center_pad_k

    torch.cuda.set_device(0)
    payload = torch.load(opts.capture, map_location="cpu", weights_only=True, mmap=True)
    if payload["q"].ndim == 4:
        length = opts.tokens or payload["q"].shape[2]
        q, k = (payload[name][:, :, :length, :].to("cuda:0")
                for name in ("q", "k"))
    elif payload["q"].ndim == 3:
        length = opts.tokens or 9482
        q, k = (payload[name][:length].to("cuda:0").transpose(0, 1).unsqueeze(0)
                for name in ("q", "k"))
    else:
        raise RuntimeError("unexpected H3 QKV capture rank")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] != 128:
        raise RuntimeError("expected H3 [B,H,N,128] capture")
    b, h, n, d = q.shape
    p = triton.cdiv(n, 128) * 128
    qm = torch.empty((b, h, p // 128, d), dtype=q.dtype, device=q.device)
    qp = torch.empty((b, h, p, d), dtype=q.dtype, device=q.device)
    _e130_group_mean_pad_q[(b, h, p // 128)](
        q, qp, qm, n, d, *q.stride(), *qp.stride(), *qm.stride(), num_warps=4)
    km = k.mean(dim=2, keepdim=True)
    kp = torch.empty((b, h, p, d), dtype=k.dtype, device=k.device)
    _e130_center_pad_k[(b, h, p // 128)](
        k, km, kp, n, d, *k.stride(), km.stride(0), km.stride(1), km.stride(3),
        *kp.stride(), num_warps=4)
    expected = torch.matmul(qm, kp.transpose(-2, -1)).contiguous()
    if expected.dtype != torch.bfloat16:
        raise RuntimeError("expected BF16 correction output")
    out = torch.empty_like(expected)
    rows = []
    for bm, bn, bk, warps in ((16, 128, 32, 4), (32, 128, 32, 4),
                              (32, 256, 32, 8), (32, 256, 64, 8)):
        def call():
            _delta_gemm[(triton.cdiv(p // 128, bm), triton.cdiv(p, bn), b * h)](
                qm, kp, out, p // 128, p,
                qm.stride(1), qm.stride(2), kp.stride(1), kp.stride(2),
                out.stride(1), out.stride(2), bm, bn, bk,
                num_warps=warps, num_stages=3)

        call()
        torch.cuda.synchronize()
        mismatch = out != expected
        diff = out.float() - expected.float()
        row = {"bm": bm, "bn": bn, "bk": bk, "warps": warps,
               "bitwise_equal": bool(torch.equal(out, expected)),
               "mismatch_fraction": float(mismatch.float().mean()),
               "nmse_vs_torch": float(diff.square().sum() /
                                      expected.float().square().sum()),
               "max_abs_diff": float(diff.abs().max()),
               "timing": measure(call)}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "timing"} |
                         {"mean_ms": row["timing"]["mean_ms"]}), flush=True)
    reference_work = torch.empty_like(expected)
    report = {"format": "e130-delta-gemm-probe-v1", "capture": str(opts.capture),
              "shape_qm": list(qm.shape), "shape_kpad": list(kp.shape),
              "shape_delta": list(expected.shape), "gpu": torch.cuda.get_device_name(),
              "torch_timing": measure(lambda: torch.matmul(
                  qm, kp.transpose(-2, -1), out=reference_work)),
              "variants": rows}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(opts.output, flush=True)


if __name__ == "__main__":
    main()
