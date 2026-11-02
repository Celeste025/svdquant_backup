#!/usr/bin/env python3
"""Separate mixed routing effect from BF16-emulator versus native Sage3 gap."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, "/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
from e113_mixed_attention import mixed_sage3_attention
from sageattn3 import sageattn3_blackwell


def nmse(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().sum() / b.square().sum()).item()


def main():
    capture = Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt")
    report = ROOT / "results/research/E113/native_vs_emulated_fp4.json"
    data = torch.load(capture, map_location="cpu", weights_only=True, mmap=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {}
    with torch.inference_mode():
        for name, begin, end in (("main", 0, 9482), ("short", 9482, 9536)):
            q, k, v = (data[key][begin:end].transpose(0, 1).unsqueeze(0).contiguous().cuda()
                       for key in ("q", "k", "v"))
            native = sageattn3_blackwell(q, k.clone(), v, is_causal=False, per_block_mean=True)
            emulate_fp4 = mixed_sage3_attention(q, k, v, threshold=0.50)
            mixed = mixed_sage3_attention(q, k, v, threshold=0.10)
            bf16 = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            result = {
                "tokens": end - begin,
                "native_vs_bf16_nmse": nmse(native, bf16),
                "emulated_fp4_vs_bf16_nmse": nmse(emulate_fp4, bf16),
                "mixed_vs_bf16_nmse": nmse(mixed, bf16),
                "emulated_fp4_vs_native_nmse": nmse(emulate_fp4, native),
                "mixed_vs_emulated_fp4_nmse": nmse(mixed, emulate_fp4),
                "mixed_vs_native_nmse": nmse(mixed, native),
            }
            results[name] = result
            print(json.dumps({name: result}, indent=2), flush=True)
    report.write_text(json.dumps({"format": "e113-native-gap-v1", "capture": str(capture),
                                  "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
