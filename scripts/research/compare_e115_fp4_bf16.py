#!/usr/bin/env python3
"""Same-H3-QKV BF16 NMSE for historical and aligned all-FP4 attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")
OLD = ROOT / "results/research/E113/native_full_attention_preflight.json"
sys.path[:0] = [str(ROOT / "scripts/research"), str(SAGE)]

from e113_native_mixed_attention import native_mixed_sage3_attention
from sageattn3 import sageattn3_blackwell


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().sum() /
                 b.float().square().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E115/fp4_vs_bf16.json")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    old = json.loads(OLD.read_text())
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[name][:9482].transpose(0, 1).unsqueeze(0)
               .contiguous().cuda() for name in ("q", "k", "v"))
    with torch.inference_mode():
        bf16 = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        official = sageattn3_blackwell(
            q, k.clone(), v, is_causal=False, per_block_mean=True)
        updated = native_mixed_sage3_attention(q, k, v, threshold=0.50)
        compatible = native_mixed_sage3_attention(
            q, k, v, threshold=0.50, match_sage3_padding_mask=True)
        report = {
            "format": "e115-fp4-vs-bf16-v1", "capture": str(CAPTURE),
            "shape": list(q.shape), "gpu": torch.cuda.get_device_name(),
            "reference": "torch.nn.functional.scaled_dot_product_attention BF16",
            "nmse_vs_bf16": {
                "historical_e113_triton_all_fp4": old["nmse_vs_bf16"]["native_triton_all_fp4"],
                "updated_triton_all_fp4_default_mask": nmse(updated, bf16),
                "updated_triton_all_fp4_official_mask": nmse(compatible, bf16),
                "official_sage3_all_fp4": nmse(official, bf16),
            },
            "updated_vs_official_nmse": nmse(updated, official),
            "compatible_vs_official_nmse": nmse(compatible, official),
            "historical_report": str(OLD),
        }
        assert abs(report["nmse_vs_bf16"]["official_sage3_all_fp4"] -
                   old["nmse_vs_bf16"]["sage3"]) < 1e-12
        assert report["compatible_vs_official_nmse"] < 1e-7
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
