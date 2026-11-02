#!/usr/bin/env python3
"""Check SM120 E4M3 MMA byte encoding against PyTorch matmul."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/sage3_native_mixed"))
import fp4attn_mixed_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E114/fp8_gemm_validation.json")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(114)
    with torch.inference_mode():
        ones = torch.ones((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
        ones_out = fp4attn_mixed_cuda.e114_debug_fp8_gemm(ones, ones)[0]
        a = torch.randn((128, 128), device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn((128, 128), device="cuda").to(torch.float8_e4m3fn)
        actual = fp4attn_mixed_cuda.e114_debug_fp8_gemm(a, b)[0]
        expected = a.float() @ b.float().T
        torch.cuda.synchronize()
        nmse = ((actual - expected).square().sum() / expected.square().sum()).item()
        report = {"format": "e114-native-fp8-gemm-v1", "gpu": torch.cuda.get_device_name(),
                  "shape": [128, 128], "dtype": "E4M3FN", "seed": 114,
                  "ones_min": ones_out.min().item(), "ones_max": ones_out.max().item(),
                  "random_nmse_vs_torch_fp32_accum": nmse,
                  "finite": bool(torch.isfinite(actual).all().item())}
        if report["ones_min"] != 128 or report["ones_max"] != 128 or \
                not report["finite"] or nmse > 1e-6:
            raise RuntimeError(f"FP8 MMA numerical gate failed: {report}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
