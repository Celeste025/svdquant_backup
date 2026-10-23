#!/usr/bin/env python3
"""Compare an isolated Sage3 build with logical and Sage3-compatible masks.

Run once per build in a fresh process: CUDA extensions with the same module
name cannot safely be switched inside one interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def nmse(x, y) -> float:
    return float((x.float() - y.float()).square().sum() /
                 y.float().square().sum())


def digest(x) -> str:
    return hashlib.sha256(
        x.contiguous().cpu().view(torch.uint16).numpy().tobytes()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Load the selected native extension and Python API before importing the
    # experimental Triton reference, which otherwise adds its default path.
    sys.path[:0] = [str(args.sage_root), str(ROOT / "scripts/research")]
    import fp4attn_cuda
    from sageattn3 import sageattn3_blackwell
    from e080_pv_contract import k_permutation
    from e113_native_mixed_attention import native_mixed_sage3_attention

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    report = {
        "format": "e115-official-sage3-mask-check-v1",
        "sage_root": str(args.sage_root.resolve()),
        "native_extension": str(Path(fp4attn_cuda.__file__).resolve()),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "capture": str(CAPTURE),
        "cases": {},
    }
    with torch.inference_mode():
        for length in (32, 54):
            physical_row = torch.arange(128)
            logical_row = k_permutation(128, "cpu")
            physical_valid = physical_row < length
            logical_valid = logical_row < length
            q, k, v = (
                payload[name][9482:9482 + length].transpose(0, 1)[:4]
                .unsqueeze(0).contiguous().to(f"cuda:{args.gpu}")
                for name in ("q", "k", "v")
            )
            original_k = k.clone()
            actual = sageattn3_blackwell(
                q, k.clone(), v, is_causal=False, per_block_mean=True)
            logical = native_mixed_sage3_attention(
                q, original_k, v, threshold=0.50)
            compatible = native_mixed_sage3_attention(
                q, original_k, v, threshold=0.50,
                match_sage3_padding_mask=True)
            assert actual.shape == logical.shape == compatible.shape
            assert all(bool(torch.isfinite(x).all()) for x in
                       (actual, logical, compatible))
            report["cases"][str(length)] = {
                "shape": list(actual.shape),
                "valid_keys_masked": int((logical_valid & ~physical_valid).sum()),
                "padding_keys_unmasked": int((~logical_valid & physical_valid).sum()),
                "sage3_vs_logical_mask_nmse": nmse(actual, logical),
                "logical_vs_sage3_nmse": nmse(logical, actual),
                "sage3_vs_compatible_mask_nmse": nmse(actual, compatible),
                "sage3_vs_logical_mask_max_abs": float(
                    (actual.float() - logical.float()).abs().max()),
                "sage3_vs_compatible_mask_max_abs": float(
                    (actual.float() - compatible.float()).abs().max()),
                "sage3_sha256": digest(actual),
                "logical_sha256": digest(logical),
                "compatible_sha256": digest(compatible),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
