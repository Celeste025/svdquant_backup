#!/usr/bin/env python3
"""Check the native Sage3 V router/FP8-128 pack against E113 tensor semantics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "third_party/sage3_native_mixed"),
    "/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe",
]
import sage3_mixed_router_cuda
from sageattn3.api import preprocess_qkv

CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")
REPORT_DIR = ROOT / "results/research/E114"


def e2m1(x: torch.Tensor) -> torch.Tensor:
    a = x.abs()
    code = torch.where(a >= 5.0, 6.0,
           torch.where(a > 3.5, 4.0,
           torch.where(a >= 2.5, 3.0,
           torch.where(a > 1.75, 2.0,
           torch.where(a >= 1.25, 1.5,
           torch.where(a > 0.75, 1.0,
           torch.where(a >= 0.25, 0.5, 0.0)))))))
    return code.copysign(x)


def reference(v: torch.Tensor, threshold: float):
    b, h, k, d = v.shape
    tiles = k // 128
    vt = v.float().view(b, h, tiles, 8, 16, d).permute(0, 1, 2, 5, 3, 4).contiguous()
    vmax16 = vt.abs().amax(dim=-1, keepdim=True)
    sf = (vmax16 / 6.0).to(torch.float8_e4m3fn).float()
    reconstructed = e2m1(torch.where(sf == 0, 0.0, vt / sf)) * sf
    diff = (reconstructed - vt).square().sum(dim=(-1, -2, -3))
    energy = vt.square().sum(dim=(-1, -2, -3))
    relsq = torch.where(energy > 0, diff / energy, 0.0)
    use_fp8 = diff >= threshold * threshold * energy
    x = vt.reshape(b, h, tiles, d, 128)
    scale = (x.abs().amax(dim=-1) / 448.0).clamp_min(1e-12)
    fp8 = (x / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (fp8.permute(0, 1, 3, 2, 4).reshape(b, h, d, k).contiguous(),
            scale.permute(0, 1, 3, 2).contiguous(), use_fp8, relsq)


def bench(fn, warmup: int = 10, repeats: int = 30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {"median_ms": samples[len(samples) // 2],
            "mean_ms": sum(samples) / len(samples),
            "min_ms": samples[0], "max_ms": samples[-1],
            "warmup": warmup, "repeats": repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=9482)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:args.tokens].transpose(0, 1)[:args.heads]
               .unsqueeze(0).contiguous().cuda() for name in ("q", "k", "v"))
    with torch.inference_mode():
        _, _, padded_v, _ = preprocess_qkv(q, k.clone(), v, per_block_mean=True)
        native = sage3_mixed_router_cuda.route_and_pack_v(padded_v, 0.10)
        ref = reference(padded_v, 0.10)
        native_fp8, native_scale, native_mask, native_relsq = native
        ref_fp8, ref_scale, ref_mask, ref_relsq = ref
        fp8_equal = torch.equal(native_fp8.float(), ref_fp8.float())
        mask_equal = torch.equal(native_mask.bool(), ref_mask)
        fp8_diff = native_fp8.float() - ref_fp8.float()
        fp8_mismatch_count = int((fp8_diff != 0).sum().item())
        fp8_max_abs_error = fp8_diff.abs().max().item()
        fp8_nmse = (fp8_diff.square().sum() / ref_fp8.float().square().sum()).item()
        scale_error = (native_scale - ref_scale).abs().max().item()
        rel_error = (native_relsq - ref_relsq).abs().max().item()
        if not mask_equal or scale_error > 1e-6 or rel_error > 1e-6:
            raise RuntimeError(f"router mismatch: FP8 {fp8_equal}, mask {mask_equal}, "
                               f"scale {scale_error}, relative sq {rel_error}")
        stats = bench(lambda: sage3_mixed_router_cuda.route_and_pack_v(padded_v, 0.10))
        row = {"format": "e114-native-sage3-router-v1",
               "shape": list(q.shape), "padded_k": padded_v.shape[2],
               "gpu": torch.cuda.get_device_name(), "threshold": 0.10,
               "fp8_group_k": 128, "fp4_group_k": 16,
               "fp8_tiles": int(native_mask.sum().item()),
               "total_tiles": native_mask.numel(),
               "mask_exact": mask_equal, "fp8_values_exact": fp8_equal,
               "fp8_mismatch_count": fp8_mismatch_count,
               "fp8_max_abs_code_error": fp8_max_abs_error,
               "fp8_code_nmse": fp8_nmse,
               "max_abs_scale_error": scale_error,
               "max_abs_relative_error_sq": rel_error,
               "router_and_fp8_pack_time": stats}
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / f"router_validation_h{args.heads}_k{args.tokens}.json").write_text(
            json.dumps(row, indent=2) + "\n")
        print(json.dumps(row, indent=2), flush=True)
        if fp8_nmse > 1e-6:
            raise RuntimeError(f"FP8 code mismatch exceeds tolerance: {fp8_nmse}")


if __name__ == "__main__":
    main()
