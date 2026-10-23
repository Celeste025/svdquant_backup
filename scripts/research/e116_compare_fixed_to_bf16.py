#!/usr/bin/env python3
"""Same-QKV BF16 NMSE for fixed pure, fixed native mixed, and Triton mixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")
PURE = ROOT / "third_party/sage3_h3_maskfix"
MIXED = ROOT / "third_party/sage3_native_mixed"


def error_sums(x: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    assert x.shape == reference.shape
    a, b = x.float(), reference.float()
    return float((a - b).square().sum()), float(b.square().sum())


def nmse(x: torch.Tensor, reference: torch.Tensor) -> float:
    numerator, denominator = error_sums(x, reference)
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E116/fixed_three_vs_bf16_gpu3.json")
    args = parser.parse_args()

    # The Triton experiment prepends its historical Sage3 path on import;
    # import the selected fixed package/extensions first to bind this process.
    sys.path[:0] = [str(PURE), str(MIXED),
                    str(ROOT / "third_party/MixAttention"),
                    str(ROOT / "scripts/research")]
    import fp4attn_cuda
    import fp4attn_mixed_maskfix_cuda
    import sage3_mixed_router_cuda
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)
    from e113_native_mixed_attention import native_mixed_sage3_attention

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    assert payload["cu_seqlens"].tolist() == [0, 9482, 9536]
    output = {
        "format": "e116-fixed-three-vs-bf16-v1",
        "capture": str(CAPTURE),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "reference": "torch.nn.functional.scaled_dot_product_attention BF16, same QKV, scale=128^-0.5, noncausal",
        "pure_extension": str(Path(fp4attn_cuda.__file__).resolve()),
        "mixed_extension": str(Path(fp4attn_mixed_maskfix_cuda.__file__).resolve()),
        "threshold": 0.10,
        "cases": {},
    }
    totals = {name: [0.0, 0.0] for name in
              ("fixed_sage3_fp4", "fixed_native_mixed", "triton_mixed")}
    with torch.inference_mode():
        for name, begin, length in (("meaningful_main", 0, 9482),
                                    ("alignment_tail", 9482, 54)):
            q, k, v = (payload[key][begin:begin + length].to(f"cuda:{args.gpu}")
                       .transpose(0, 1).unsqueeze(0)
                       for key in ("q", "k", "v"))
            assert q.shape == k.shape == v.shape == (1, 56, length, 128)
            bf16 = F.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5)
            qp, kp, vp, ds = preprocess_qkv(q, k.clone(), v, True)
            q4 = scale_and_quant_fp4(qp)
            k4 = scale_and_quant_fp4_permute(kp)
            v4 = scale_and_quant_fp4_transpose(vp)
            packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                      ds, length, None, 128 ** -0.5, False, True, True)
            pure = fp4attn_cuda.fwd(*packet)[0][:, :, :length].contiguous()
            v8, s8, route, _ = sage3_mixed_router_cuda.route_and_pack_v(vp, 0.10)
            native = fp4attn_mixed_maskfix_cuda.fwd_mixed(
                *packet, v8, s8, route)[0][:, :, :length].contiguous()
            triton = native_mixed_sage3_attention(
                q, k, v, threshold=0.10)
            arms = {"fixed_sage3_fp4": pure,
                    "fixed_native_mixed": native,
                    "triton_mixed": triton}
            assert all(bool(torch.isfinite(t).all()) for t in
                       (bf16, *arms.values()))
            errors = {}
            for arm, value in arms.items():
                numerator, denominator = error_sums(value, bf16)
                errors[arm] = {"nmse": numerator / denominator,
                               "squared_error_sum": numerator,
                               "bf16_energy_sum": denominator}
                totals[arm][0] += numerator
                totals[arm][1] += denominator
            case = {"qkv_shape": list(q.shape), "bf16_output_energy":
                    float(bf16.float().square().sum()),
                    "fp8_tiles": int(route.sum()), "total_v_tiles": route.numel(),
                    "errors_vs_bf16": errors,
                    "triton_vs_native_mixed_nmse": nmse(triton, native),
                    "native_mixed_vs_pure_nmse": nmse(native, pure)}
            output["cases"][name] = case
            print(name, json.dumps(case), flush=True)
    output["both_segments_aggregate_nmse"] = {
        name: numerator / denominator
        for name, (numerator, denominator) in totals.items()
    }
    main_errors = output["cases"]["meaningful_main"]["errors_vs_bf16"]
    pure_nmse = main_errors["fixed_sage3_fp4"]["nmse"]
    output["main_relative_nmse_reduction_vs_pure_percent"] = {
        arm: 100 * (pure_nmse - main_errors[arm]["nmse"]) / pure_nmse
        for arm in ("fixed_native_mixed", "triton_mixed")
    }
    assert output["cases"]["meaningful_main"]["triton_vs_native_mixed_nmse"] < 1e-7
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("main_relative_reduction_percent",
          output["main_relative_nmse_reduction_vs_pure_percent"], flush=True)


if __name__ == "__main__":
    main()
