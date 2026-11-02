#!/usr/bin/env python3
"""Check whether sparse P8V8-minus-P4V4 correction can match native mixed."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def nmse(a, b):
    a, b = a.float(), b.float()
    return float((a - b).square().sum() / b.square().sum())


def e2m1_round(x):
    ax = x.abs()
    mag = torch.where(ax >= 5, 6., torch.where(ax > 3.5, 4.,
          torch.where(ax >= 2.5, 3., torch.where(ax > 1.75, 2.,
          torch.where(ax >= 1.25, 1.5, torch.where(ax > 0.75, 1.,
          torch.where(ax >= 0.25, 0.5, 0.)))))))
    return torch.copysign(mag, x)


def main():
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed"),
                    str(ROOT / "scripts/research")]
    import fp4attn_lse_cuda
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda
    from e119_sparse_v_correction import pack_selected_delta_v
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:128].to("cuda:0").transpose(0, 1)[:4]
               .unsqueeze(0) for name in ("q", "k", "v"))
    with torch.inference_mode():
        qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, 128, None, 128 ** -0.5, False, True, True)
        v8, s8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
        route = torch.ones_like(route)
        pure, lse = fp4attn_lse_cuda.fwd(*packet)[:2]
        mixed = fp4attn_mixed_maskfix_cuda.fwd_mixed(
            *packet, v8, s8, route)[0]
        delta = torch.empty_like(vp)
        pack_selected_delta_v[(1, 4, 1)](vp, v8, s8, route, delta, 4, 128, 128, 1)
        v8_dequant = v8.float().transpose(-1, -2) * s8.float().squeeze(-1)[:, :, None, :]
        v4_dequant = v8_dequant - delta.float()
        scores = torch.matmul(q.float(), kp.float().transpose(-1, -2)) * (128 ** -0.5)
        tile_max = scores.amax(dim=-1, keepdim=True)
        p8_code = (448.0 * torch.exp(scores - tile_max)).to(torch.float8_e4m3fn).float()
        p8 = p8_code * torch.exp(tile_max - lse[..., None]) / 448.0
        group_score = scores.reshape(1, 4, 128, 8, 16)
        group_max = group_score.amax(dim=-1, keepdim=True)
        p4_code = e2m1_round(6.0 * torch.exp(group_score - group_max))
        p4 = (p4_code * torch.exp(group_max - lse[..., None, None]) / 6.0).reshape_as(scores)
        approx = pure.float() + torch.matmul(p8, v8_dequant) - torch.matmul(p4, v4_dequant)
        ref = torch.matmul(torch.softmax(
            torch.matmul(q.float(), k.float().transpose(-1, -2)) * (128 ** -0.5),
            dim=-1), v.float())
        report = {
            "format": "e119-probability-correction-probe-v1",
            "shape": list(q.shape),
            "pure_vs_bf16_nmse": nmse(pure, ref),
            "native_mixed_vs_bf16_nmse": nmse(mixed, ref),
            "approx_correction_vs_bf16_nmse": nmse(approx, ref),
            "approx_correction_vs_native_mixed_nmse": nmse(approx, mixed),
            "approx_correction_vs_pure_nmse": nmse(approx, pure),
            "p4_range": [float(p4.min()), float(p4.max())],
            "p8_range": [float(p8.min()), float(p8.max())],
            "delta_v_rms": float(delta.float().square().mean().sqrt()),
        }
    out = ROOT / "results/research/E119/probability_correction_control128.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
