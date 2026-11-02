#!/usr/bin/env python3
"""Compare mixed Triton core with a 128-KV-tile PyTorch reference on real H3 QKV."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
sys.path.insert(0, str(ROOT / "third_party/MixAttention"))
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(SAGE))
from pv_hadamard.quant import nvfp4_qdq, nvfp4_qdq_v, fp8_qdq_v, quant_p, fp8_quant_p
from sageattn3.api import preprocess_qkv
from e113_mixed_attention import mixed_sage3_attention


def reference(q, k, v, threshold, *, fp8_group=16, reverse_tiles=False):
    qpp, kpp, vpp, delta = preprocess_qkv(q, k.clone(), v, per_block_mean=True)
    length = q.shape[2]
    padded = qpp.shape[2]
    scale = q.shape[-1] ** -0.5
    q4 = nvfp4_qdq(qpp[0].float()).bfloat16()
    k4 = nvfp4_qdq(kpp[0].float()).bfloat16()
    vf = vpp[0].float()
    v4 = nvfp4_qdq_v(vf)
    v8 = fp8_qdq_v(vf, group=fp8_group)
    d = delta[0].float() * scale
    if padded > length:
        d[:, :, length:] = -float("inf")
    h = q.shape[1]
    m = torch.full((h, padded), -float("inf"), device=q.device)
    den = torch.zeros_like(m)
    out = torch.zeros((h, padded, q.shape[-1]), device=q.device)
    route = []
    starts = (range(padded - 128, -1, -128) if reverse_tiles
              else range(0, padded, 128))
    for start in starts:
        sl = slice(start, start + 128)
        raw = vf[:, sl]
        err = (v4[:, sl] - raw).square().sum((1, 2))
        energy = raw.square().sum((1, 2))
        use_fp4 = err < (threshold ** 2) * energy
        route.append(use_fp4.cpu().tolist())
        logits = torch.bmm(q4, k4[:, sl].transpose(1, 2), out_dtype=torch.float32) * scale
        logits += d[:, torch.arange(padded, device=q.device) // 128, sl]
        m_next = torch.maximum(m, logits.amax(-1))
        alpha = torch.exp(m - m_next)
        prob = torch.exp(logits - m_next[:, :, None])
        den = den * alpha + prob.sum(-1)
        p4 = quant_p(prob)
        p8 = fp8_quant_p(prob, group=fp8_group)
        pq = torch.where(use_fp4[:, None, None], p4, p8).bfloat16()
        vq = torch.where(use_fp4[:, None, None], v4[:, sl], v8[:, sl]).bfloat16()
        out = out * alpha[:, :, None] + torch.bmm(pq, vq, out_dtype=torch.float32)
        m = m_next
    return (out / den[:, :, None])[:, :length].unsqueeze(0).bfloat16(), route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    with torch.no_grad():
        for seq_name, begin, n in (("main", 0, 1024),
                                   ("main_full", 0, 9482),
                                   ("short", 9482, 54)):
            for head in ((0,) if seq_name == "main_full" else (0, 4)):
                q, k, v = (data[key][begin:begin+n, head].unsqueeze(0).unsqueeze(0).bfloat16().cuda()
                           for key in ("q", "k", "v"))
                thresholds = (0.10,) if seq_name == "main_full" else (0.001, 0.10, 0.50)
                for threshold in thresholds:
                    ref, route = reference(q, k, v, threshold)
                    tri = mixed_sage3_attention(q, k, v, threshold=threshold)
                    diff = tri.float() - ref.float()
                    nmse = (diff.square().sum() / ref.float().square().sum()).item()
                    max_abs = diff.abs().max().item()
                    assert torch.isfinite(tri).all().item() and nmse < 1e-6, (seq_name, head, threshold, nmse)
                    row = {"sequence": seq_name, "head": head, "length": n,
                           "threshold": threshold, "fp4_tiles": sum(x[0] for x in route),
                           "total_tiles": len(route), "nmse": nmse,
                           "rel_l2": math.sqrt(nmse), "max_abs": max_abs,
                           "finite": bool(torch.isfinite(tri).all())}
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    if seq_name == "main" and head == 4 and threshold == 0.10:
                        assert 0 < row["fp4_tiles"] < row["total_tiles"], "mixed route not exercised"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
