#!/usr/bin/env python3
"""Validate selected-only FP8 V packing and compare it with dense packing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def nmse(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    return float((x - y).square().sum() / y.square().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E117/sparse_router_gpu0.json")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed"),
                    str(ROOT / "scripts/research")]
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda as dense_router
    import sage3_mixed_router_sparse_cuda as sparse_router
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)
    from e117_native_mixed_cost_split import time_cuda

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    report = {"format": "e117-sparse-router-v1", "capture": str(CAPTURE),
              "gpu": torch.cuda.get_device_name(args.gpu), "cases": {},
              "timing": {}}
    with torch.inference_mode():
        for name, start, length in (("main_9482", 0, 9482),
                                    ("tail_54", 9482, 54)):
            q, k, v = (payload[key][start:start + length]
                       .to(f"cuda:{args.gpu}").transpose(0, 1).unsqueeze(0)
                       for key in ("q", "k", "v"))
            qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
            q4 = scale_and_quant_fp4(qp)
            k4 = scale_and_quant_fp4_permute(kp)
            v4 = scale_and_quant_fp4_transpose(vp)
            packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                      ds, length, None, 128 ** -0.5, False, True, True)
            dense_v8, dense_scale, dense_mask, dense_error = \
                dense_router.route_and_pack_v(vp, 0.10)
            sparse_v8, sparse_scale, sparse_mask, sparse_error = \
                sparse_router.route_and_pack_v(vp, 0.10)
            assert torch.equal(dense_mask, sparse_mask)
            assert torch.equal(dense_error, sparse_error)
            b, h, d, k_padded = dense_v8.shape
            tiles = k_padded // 128
            selected_scale = dense_mask[:, :, None, :].expand(b, h, d, tiles)
            selected_value = dense_mask[:, :, None, :, None].expand(
                b, h, d, tiles, 128)
            scale_equal = torch.equal(dense_scale[selected_scale.bool()],
                                      sparse_scale[selected_scale.bool()])
            value_equal = torch.equal(
                dense_v8.view(torch.uint8).reshape(b, h, d, tiles, 128)[
                    selected_value.bool()],
                sparse_v8.view(torch.uint8).reshape(b, h, d, tiles, 128)[
                    selected_value.bool()])
            assert scale_equal and value_equal
            dense_output = fp4attn_mixed_maskfix_cuda.fwd_mixed(
                *packet, dense_v8, dense_scale, dense_mask)[0][:, :, :length]
            sparse_output = fp4attn_mixed_maskfix_cuda.fwd_mixed(
                *packet, sparse_v8, sparse_scale, sparse_mask)[0][:, :, :length]
            assert torch.equal(dense_output, sparse_output)
            report["cases"][name] = {
                "selected_tiles": int(dense_mask.sum()),
                "total_tiles": dense_mask.numel(),
                "selected_scale_bitwise_equal": scale_equal,
                "selected_fp8_code_bitwise_equal": value_equal,
                "attention_output_bitwise_equal": True,
                "attention_output_nmse": nmse(dense_output, sparse_output),
            }
            if name == "main_9482":
                for scope, fn in (
                    ("router_dense", lambda: dense_router.route_and_pack_v(vp, 0.10)),
                    ("router_sparse", lambda: sparse_router.route_and_pack_v(vp, 0.10))
                ):
                    report["timing"][scope] = time_cuda(fn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": report["cases"],
                      "timing_ms": {name: row["mean_ms"] for name, row
                                    in report["timing"].items()}}, indent=2))


if __name__ == "__main__":
    main()
