#!/usr/bin/env python3
"""Diagnose the non-monotone E142 clap/block49 sparse-attention error.

The key comparison keeps the DraftMap support fixed at each sparsity and
separates (a) BF16 support truncation from (b) low-precision arithmetic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/data1/models/svdquant-wjq/research/20261031/E142_attention_tradeoff")
OUT = ROOT / "results/research/E143"
sys.path.insert(0, str(ROOT / "third_party/anemoi-code"))

from anemoi import (  # noqa: E402
    NVFP4Calibration,
    QuantConfig,
    SparseConfig,
    VisualLayout,
    anemoi_attention,
)


def stats(x: torch.Tensor) -> dict[str, float]:
    x = x.float()
    a = x.abs().flatten()
    # torch.quantile rejects this ~150M-element tensor.  Use a deterministic
    # uniform sample only for percentiles; all moments/extrema below are exact.
    qsample = a[:: max(1, math.ceil(a.numel() / 2_000_000))]
    qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999], device=x.device)
    v = torch.quantile(qsample, qs).cpu().tolist()
    return {
        "finite_fraction": float(torch.isfinite(x).float().mean()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": float(x.square().mean().sqrt()),
        "absmax": float(a.max()),
        "percentile_sample_count": int(qsample.numel()),
        **{f"abs_p{100*q:g}": float(z) for q, z in zip(qs.cpu().tolist(), v)},
    }


def error_terms(out: torch.Tensor, ref: torch.Tensor, sparse_ref: torch.Tensor) -> dict:
    # Restrict to video query outputs; all three tensors are BSHD.
    e_sparse = (sparse_ref - ref).float()
    e_quant = (out - sparse_ref).float()
    e_total = (out - ref).float()
    denom = ref.float().square().sum(dtype=torch.float64)
    sparse = e_sparse.square().sum(dtype=torch.float64)
    quant = e_quant.square().sum(dtype=torch.float64)
    total = e_total.square().sum(dtype=torch.float64)
    cross = 2.0 * (e_sparse.double() * e_quant.double()).sum()
    cos = (e_sparse.double() * e_quant.double()).sum() / (
        sparse.sqrt() * quant.sqrt() + 1e-30
    )
    head_num = e_total.square().sum(dim=(0, 1, 3), dtype=torch.float64)
    head_den = ref.float().square().sum(dim=(0, 1, 3), dtype=torch.float64)
    head_nmse = head_num / head_den
    top = torch.topk(head_nmse, k=5)
    return {
        "nmse": float(total / denom),
        "sparsity_nmse": float(sparse / denom),
        "conditional_quant_nmse": float(quant / denom),
        "cross_nmse": float(cross / denom),
        "identity_residual": float((total - sparse - quant - cross).abs() / denom),
        "sparse_quant_error_cosine": float(cos),
        "output_absmax": float(out.abs().max()),
        "output_finite_fraction": float(torch.isfinite(out).float().mean()),
        "head_nmse_median": float(head_nmse.median()),
        "head_nmse_max": float(head_nmse.max()),
        "head_nmse_top5": [
            {"head": int(i), "nmse": float(z)}
            for z, i in zip(top.values.cpu(), top.indices.cpu())
        ],
    }


@torch.inference_mode()
def main() -> None:
    sample = torch.load(DATA / "clap_s10_b49.pt", map_location="cpu", weights_only=True, mmap=True)
    q, k, v = [sample[name].cuda().transpose(1, 2).contiguous() for name in ("q", "k", "v")]
    prefix = int(sample["prefix_tokens"])
    layout = VisualLayout(tuple(sample["video_grid"]), prefix)
    scale_json = json.loads(
        (ROOT / "third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json").read_text()
    )["scales"]
    layer = int(sample["layer"])
    calibration = NVFP4Calibration(
        *(float(scale_json[key][layer]) for key in ("q_tensor_scale", "k_tensor_scale", "v_tensor_scale"))
    )
    ref_all = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2).contiguous()
    ref = ref_all[:, prefix:]

    def run(skip: float, nvfp4: float, int8: float, fp16: float,
            cal: NVFP4Calibration | None = None) -> torch.Tensor:
        out = anemoi_attention(
            q, k, v,
            layout=layout,
            layer=layer,
            sparse_config=SparseConfig(query_block_size=64, sparsity_ratio=skip, maxpool_weight=0.2),
            quant_config=QuantConfig(
                nvfp4_ratio=nvfp4,
                int8_ratio=int8,
                fp16_ratio=fp16,
                prefix_kv_precision="int8",
                # Prefix-query output is discarded in this video-only audit.
                # The executor requires an active video INT8 phase to use the
                # native prefix-query INT8 kernel.
                prefix_query_precision="int8" if int8 else "fp16",
            ),
            calibration=(cal or calibration) if nvfp4 else None,
        )
        return out[:, prefix:].contiguous()

    report: dict = {
        "format": "e143-clap-b49-anomaly-diagnostic-v1",
        "sample": "clap_s10_b49",
        "shape_bshd": list(q.shape),
        "video_grid": list(sample["video_grid"]),
        "prefix_tokens": prefix,
        "layer": layer,
        "calibration": dict(zip(("q", "k", "v"), calibration.scales)),
        "reference": stats(ref),
        "inputs": {},
        "arms": {},
        "repeatability": {},
        "calibration_sweep": {},
        "component_scale_sweep": {},
    }
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        report["inputs"][name] = {
            "all": stats(tensor),
            "prefix": stats(tensor[:, :prefix]),
            "video": stats(tensor[:, prefix:]),
        }

    # The FP16 arm at each skip defines the exact same top-k support without
    # low-precision video arithmetic. Quant arms share that support because
    # routing is computed from the same original pooled Q/K.
    for skip in (0.0, 0.3, 0.5, 0.7, 0.8):
        sparse_ref = run(skip, 0.0, 0.0, 1.0)
        sparse_key = f"skip{round(skip * 100):02}_fp16"
        report["arms"][sparse_key] = error_terms(sparse_ref, ref, sparse_ref)
        specs = {
            "pure_fp4": (1.0, 0.0, 0.0),
            "mix_i025": (0.75, 0.25, 0.0),
            "mix_i050": (0.50, 0.50, 0.0),
            "pure_int8": (0.0, 1.0, 0.0),
        }
        for label, ratios in specs.items():
            out = run(skip, *ratios)
            key = f"skip{round(skip * 100):02}_{label}"
            report["arms"][key] = error_terms(out, ref, sparse_ref)
            if skip in (0.0, 0.8) and label in ("pure_fp4", "mix_i025", "pure_int8"):
                again = run(skip, *ratios)
                report["repeatability"][key] = {
                    "exact_equal": bool(torch.equal(out, again)),
                    "max_abs_difference": float((out - again).abs().max()),
                }
            del out
        del sparse_ref

    # Scale sensitivity at dense support.  If the anomaly is a bad static
    # tensor-scale choice, modest scale changes should collapse the error.
    dense_sparse_ref = run(0.0, 0.0, 0.0, 1.0)
    for factor in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        cal = NVFP4Calibration(*(factor * x for x in calibration.scales))
        for label, ratios in {
            "pure_fp4": (1.0, 0.0, 0.0),
            "mix_i025": (0.75, 0.25, 0.0),
        }.items():
            out = run(0.0, *ratios, cal=cal)
            report["calibration_sweep"][f"factor{factor:g}_{label}"] = error_terms(
                out, ref, dense_sparse_ref
            )
    del dense_sparse_ref

    dense_sparse_ref = run(0.0, 0.0, 0.0, 1.0)
    for component in range(3):
        for factor in (0.5, 0.75, 1.25, 1.5):
            values = list(calibration.scales)
            values[component] *= factor
            cal = NVFP4Calibration(*values)
            out = run(0.0, 0.75, 0.25, 0.0, cal=cal)
            name = ("q", "k", "v")[component]
            report["component_scale_sweep"][f"{name}_factor{factor:g}"] = error_terms(
                out, ref, dense_sparse_ref
            )
    del dense_sparse_ref

    # Sanity check the energy identity and finite output for every arm.
    for key, row in report["arms"].items():
        assert row["output_finite_fraction"] == 1.0, key
        assert row["identity_residual"] < 1e-6, (key, row["identity_residual"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clap_s10_b49_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
