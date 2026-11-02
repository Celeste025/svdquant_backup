#!/usr/bin/env python3
"""Probe whether raw-K GEMM plus a mean correction can replace centered-K GEMM."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research"),
                str(ROOT / "third_party/sage3_h3_maskfix")]
from bench_h3_sage_real_gpu0 import metrics

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt")
OUTPUT = ROOT / "results/research/E130/raw_k_correction_probe_gpu2.json"


def main() -> None:
    import sageattn3.api as api

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    n = 9482
    p = ((n + 127) // 128) * 128
    q, k, v = [payload[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
               for name in ("q", "k", "v")]
    with torch.inference_mode():
        qcenter, qm = api.triton_group_mean(F.pad(q, (0, 0, 0, p - n)).contiguous())
        kmean = k.mean(dim=2, keepdim=True)
        kcenter = k.clone().sub_(kmean)
        kcenter_pad = F.pad(kcenter, (0, 0, 0, p - n)).contiguous()
        kraw_pad = F.pad(k, (0, 0, 0, p - n)).contiguous()
        vpad = F.pad(v, (0, 0, 0, p - n)).contiguous()
        baseline_delta = torch.matmul(qm, kcenter_pad.transpose(-2, -1)).float()
        raw_delta = torch.matmul(qm, kraw_pad.transpose(-2, -1)).float()
        mean_delta = torch.matmul(qm, kmean.transpose(-2, -1)).float()
        decomposed_delta = raw_delta - mean_delta
        decomposed_delta[..., n:] = 0
        qpack = api.scale_and_quant_fp4(qcenter)
        kpack = api.scale_and_quant_fp4_permute(kcenter_pad)
        vpack = api.scale_and_quant_fp4_transpose(vpad)
        def run(delta):
            return api.blockscaled_fp4_attn(qpack, kpack, vpack, delta, n,
                                            is_causal=False, per_block_mean=True,
                                            is_bf16=True)[0][:, :, :n, :].contiguous()
        baseline = run(baseline_delta)
        candidate = run(decomposed_delta.contiguous())
        delta_diff = decomposed_delta[..., :n] - baseline_delta[..., :n]
        report = {"format": "e130-raw-k-correction-probe-v1", "capture": str(CAPTURE),
                  "tokens": n, "definition": "BF16 qm@K minus BF16 qm@mean, then FP32 subtraction",
                  "delta_nmse": float(delta_diff.square().sum() /
                                      baseline_delta[..., :n].square().sum()),
                  "delta_max_abs_difference": float(delta_diff.abs().max()),
                  "output_vs_centered_k": metrics(candidate, baseline),
                  "all_finite": bool(torch.isfinite(candidate).all())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
