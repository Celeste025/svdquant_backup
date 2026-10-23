#!/usr/bin/env python3
"""Audit corrected E113 Triton FP4 against official Sage3 on real H3 QKV.

This is a numerical diagnostic, not a performance benchmark. Historical E113
and E114 reports are left untouched; the output is an E115 JSON artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")
sys.path[:0] = [str(ROOT / "third_party/sage3_native_mixed"),
                str(ROOT / "third_party/MixAttention"),
                str(ROOT / "scripts/research"), str(SAGE)]

import fp4attn_cuda
import fp4attn_mixed_cuda
import sage3_mixed_router_cuda
from e080_pv_contract import decode_packed, k_permutation, logical_scales
from e113_native_mixed_attention import _e2m1_code, native_mixed_sage3_attention
from pv_hadamard.quant import nvfp4_qdq, nvfp4_qdq_v
from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                           scale_and_quant_fp4_permute,
                           scale_and_quant_fp4_transpose)


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().sum() /
                 b.float().square().sum())


@triton.jit
def _pack_probe(X, C, S, COLS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, COLS)
    group = tl.reshape(tl.load(X + row * COLS + col).to(tl.float32),
                       (COLS // 16, 16))
    scale = (tl.max(tl.abs(group), 1) / 6.).to(tl.float8e4nv)
    scale32 = scale.to(tl.float32)
    inverse = tl.where(scale32 == 0, 0., 1. / scale32)
    code = _e2m1_code(group * inverse[:, None])
    tl.store(C + row * COLS + col, tl.reshape(code, (COLS,)))
    tl.store(S + row * (COLS // 16) + tl.arange(0, COLS // 16), scale)


def triton_packet_audit(x: torch.Tensor, official_codes: torch.Tensor,
                        official_scales: torch.Tensor, name: str) -> dict:
    if name == "v":
        x = x.transpose(-2, -1).contiguous()
    else:
        x = x.contiguous()
    _, heads, rows, cols = x.shape
    code = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
    scale = torch.empty((*x.shape[:-1], cols // 16), device=x.device,
                        dtype=torch.float8_e4m3fn)
    _pack_probe[(heads * rows,)](x, code, scale, cols)
    actual_code = torch.stack((official_codes & 15, official_codes >> 4),
                              -1).flatten(-2)
    actual_scale = logical_scales(official_scales)
    if name == "k":
        inverse = k_permutation(rows, x.device).argsort()
        actual_code = actual_code.index_select(-2, inverse)
        actual_scale = actual_scale.index_select(-2, inverse)
    return {"code_mismatches": int((code != actual_code).sum()),
            "scale_mismatches": int((scale.view(torch.uint8) !=
                                     actual_scale.view(torch.uint8)).sum()),
            "elements": code.numel()}


def qkv_packet_audit(q: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> dict:
    qp, kp, vp, _ = preprocess_qkv(q, k.clone(), v, True)
    records = {}
    for name, x, pack, simulate in (
        ("q", qp, scale_and_quant_fp4, nvfp4_qdq),
        ("k", kp, scale_and_quant_fp4_permute, nvfp4_qdq),
        ("v", vp, scale_and_quant_fp4_transpose, nvfp4_qdq_v),
    ):
        codes, scales = pack(x)
        actual = decode_packed(codes, scales, name)
        expected = simulate(x.float())
        records[name] = {
            "nmse_official_pack_vs_rne_reference": nmse(expected, actual),
            "mismatched_decoded_elements": int((actual != expected).sum()),
            "elements": actual.numel(),
            "max_abs_difference": float((actual - expected).abs().max()),
            "triton_actual_code": triton_packet_audit(x, codes, scales, name),
        }
    return records


def output_audit(q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor) -> dict:
    valid = q.shape[2]
    qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
    q4 = scale_and_quant_fp4(qp)
    k4 = scale_and_quant_fp4_permute(kp)
    v4 = scale_and_quant_fp4_transpose(vp)
    packed = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1], ds,
              valid, None, 128 ** -0.5, False, True, True)
    off = fp4attn_cuda.fwd(*packed)[0][:, :, :valid].contiguous()
    v8, s8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
    native = fp4attn_mixed_cuda.fwd_mixed(*packed, v8, s8, route)[0][
        :, :, :valid].contiguous()
    tri4 = native_mixed_sage3_attention(q, k, v, threshold=0.50)
    tri = native_mixed_sage3_attention(q, k, v, threshold=0.10)
    tri4_compat = native_mixed_sage3_attention(
        q, k, v, threshold=0.50, match_sage3_padding_mask=True)
    tri_compat = native_mixed_sage3_attention(
        q, k, v, threshold=0.10, match_sage3_padding_mask=True)
    dn = native.float() - off.float()
    dt = tri.float() - tri4.float()
    dt_compat = tri_compat.float() - tri4_compat.float()
    delta_norm = dn.square().sum().sqrt() * dt.square().sum().sqrt()
    compat_norm = dn.square().sum().sqrt() * dt_compat.square().sum().sqrt()
    return {
        "selected_fp8_tiles": int(route.sum()), "total_v_tiles": route.numel(),
        "triton_all_fp4_vs_official_fp4_nmse": nmse(tri4, off),
        "triton_all_fp4_compat_vs_official_fp4_nmse": nmse(tri4_compat, off),
        "native_mixed_vs_official_fp4_nmse": nmse(native, off),
        "triton_mixed_vs_triton_all_fp4_nmse": nmse(tri, tri4),
        "triton_mixed_compat_vs_triton_all_fp4_compat_nmse": nmse(
            tri_compat, tri4_compat),
        "triton_mixed_compat_vs_native_mixed_nmse": nmse(tri_compat, native),
        "triton_mixed_vs_native_mixed_nmse": nmse(tri, native),
        "mixed_delta_difference_nmse": float((dn - dt).square().sum() /
                                             off.float().square().sum()),
        "mixed_delta_cosine": (float((dn * dt).sum() / delta_norm)
                               if float(delta_norm) > 0 else None),
        "mixed_delta_compat_cosine": (
            float((dn * dt_compat).sum() / compat_norm)
            if float(compat_norm) > 0 else None),
        "all_outputs_finite": all(bool(torch.isfinite(x).all())
                                  for x in (off, native, tri4, tri,
                                            tri4_compat, tri_compat)),
        "triton_native_mma_check": native_mixed_sage3_attention.last_mma_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E115/fp4_alignment_reverse_kv.json")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    data = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    report = {"format": "e115-sage3-fp4-alignment-v1", "capture": str(CAPTURE),
              "gpu": torch.cuda.get_device_name(), "cases": {}}
    with torch.inference_mode():
        for name, begin, length, heads in (
            ("main_1024_h4", 0, 1024, 4),
            ("main_9482_h56", 0, 9482, 56),
            ("short_54_h4", 9482, 54, 4),
        ):
            q, k, v = (data[key][begin:begin + length].transpose(0, 1)
                       [:heads].unsqueeze(0).contiguous().cuda()
                       for key in ("q", "k", "v"))
            case = {"shape": list(q.shape)}
            if name == "main_1024_h4":
                case["qkv_pack"] = qkv_packet_audit(q, k, v)
            case["output"] = output_audit(q, k, v)
            report["cases"][name] = case
            print(name, json.dumps(case), flush=True)
    assert all(c["output"]["all_outputs_finite"] for c in report["cases"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
