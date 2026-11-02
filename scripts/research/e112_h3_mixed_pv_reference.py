#!/usr/bin/env python3
"""Real-H3 PyTorch reference for V-guided mixed FP4/FP8 P/V attention.

This is a numerical study, not a native mixed-precision speed benchmark.
The FP8 arm follows MixAttention's Python prototype (per-16 dynamic scales).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/MixAttention"))
from pv_hadamard.quant import nvfp4_qdq, nvfp4_qdq_v, fp8_qdq_v, quant_p, fp8_quant_p


def tensor_nmse(y: torch.Tensor, ref: torch.Tensor) -> float:
    return ((y - ref).square().sum() / ref.square().sum()).item()


def study_sequence(q_cpu, k_cpu, v_cpu, scale, query_count, threshold, device):
    assert q_cpu.shape == k_cpu.shape == v_cpu.shape
    length, heads, dim = q_cpu.shape
    assert dim == 128 and heads == 56
    pad = (-length) % 128
    full = length + pad
    tiles = full // 128
    q = q_cpu.permute(1, 0, 2).contiguous().to(device=device, dtype=torch.float32)
    k = k_cpu.permute(1, 0, 2).contiguous().to(device=device, dtype=torch.float32)
    v = v_cpu.permute(1, 0, 2).contiguous().to(device=device, dtype=torch.float32)

    # Query positions are deterministic and span the whole real sequence.
    qi = torch.linspace(0, length - 1, min(query_count, length), device=device)
    qi = qi.round().long().unique(sorted=True)
    qs = q.index_select(1, qi)
    with torch.no_grad():
        ref = torch.nn.functional.scaled_dot_product_attention(
            qs, k, v, scale=scale, is_causal=False)

        # Match Sage3's ordering: center valid K first, then pad; Q mean is
        # computed per padded 128-query block. No Q/K route changes across arms.
        kc = k - k.mean(dim=1, keepdim=True)
        if pad:
            qp = torch.nn.functional.pad(q, (0, 0, 0, pad))
            kp = torch.nn.functional.pad(kc, (0, 0, 0, pad))
            vp = torch.nn.functional.pad(v, (0, 0, 0, pad))
        else:
            qp, kp, vp = q, kc, v
        qm = qp.reshape(heads, tiles, 128, dim).mean(dim=2)
        q_centered = qp - qm.repeat_interleave(128, dim=1)
        q4 = nvfp4_qdq(q_centered).index_select(1, qi)
        k4 = nvfp4_qdq(kp)
        qmeans = qm.index_select(1, qi // 128)
        v4 = nvfp4_qdq_v(vp)
        v8 = fp8_qdq_v(vp)

        delta = (v4 - vp).reshape(heads, tiles, 128, dim)
        sq_error = delta.square().sum(dim=(2, 3))
        energy = vp.reshape(heads, tiles, 128, dim).square().sum(dim=(2, 3))
        rel_l2 = (sq_error / energy.clamp_min(1e-30)).sqrt()
        fp8_mask = rel_l2 >= threshold
        fp8_mask &= energy > 0

        # Also test a strict 1%-tile budget, since that was the original
        # desired mix ratio. Its threshold is derived solely from V errors.
        eligible = energy > 0
        n_budget = max(1, math.ceil(0.01 * int(eligible.sum().item())))
        top_scores = rel_l2.masked_fill(~eligible, -float("inf")).flatten()
        budget_mask = torch.zeros_like(top_scores, dtype=torch.bool)
        budget_mask[top_scores.topk(n_budget).indices] = True
        budget_mask = budget_mask.reshape(heads, tiles)

        outputs = {
            name: torch.zeros(heads, qi.numel(), dim, device=device)
            for name in ("qk_only", "all_fp4", "all_fp8", "mixed_0p10", "top1pct_fp8")
        }
        m = torch.full((heads, qi.numel()), -float("inf"), device=device)
        denom = torch.zeros_like(m)
        for t in range(tiles):
            sl = slice(t * 128, (t + 1) * 128)
            logits = (
                torch.bmm(q4, k4[:, sl].transpose(1, 2))
                + torch.bmm(qmeans, kp[:, sl].transpose(1, 2))
            ) * scale
            valid = max(0, min(128, length - t * 128))
            if valid < 128:
                logits[:, :, valid:] = -float("inf")
            m_new = torch.maximum(m, logits.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(logits - m_new.unsqueeze(-1))
            denom = denom * alpha + p.sum(dim=-1)
            p4 = quant_p(p)
            p8 = fp8_quant_p(p)
            route = fp8_mask[:, t, None, None]
            budget_route = budget_mask[:, t, None, None]
            operands = {
                "qk_only": (p, vp[:, sl]),
                "all_fp4": (p4, v4[:, sl]),
                "all_fp8": (p8, v8[:, sl]),
                "mixed_0p10": (torch.where(route, p8, p4),
                                torch.where(route, v8[:, sl], v4[:, sl])),
                "top1pct_fp8": (torch.where(budget_route, p8, p4),
                                 torch.where(budget_route, v8[:, sl], v4[:, sl])),
            }
            for name, (p_arm, v_arm) in operands.items():
                outputs[name] = outputs[name] * alpha.unsqueeze(-1) + torch.bmm(p_arm, v_arm)
            m = m_new
        outputs = {name: out / denom.unsqueeze(-1) for name, out in outputs.items()}
        assert all(torch.isfinite(out).all().item() for out in outputs.values())
        metrics = {name: {"nmse": tensor_nmse(out, ref),
                          "rel_l2": math.sqrt(tensor_nmse(out, ref))}
                   for name, out in outputs.items()}
        # The comparison to the same-QK unquantized-P/V arm isolates PV damage.
        metrics_vs_qk_only = {
            name: {"nmse": tensor_nmse(out, outputs["qk_only"])}
            for name, out in outputs.items() if name != "qk_only"
        }
        ref_energy = ref.square().sum().item()
        assert ref_energy > 0
        thresholds = {}
        for th in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
            th_mask = (rel_l2 >= th) & eligible
            thresholds[f"{th:.2f}"] = int(th_mask.sum().item()) / int(eligible.sum().item())
        result = {
            "length": length, "padded_length": full, "heads": heads,
            "kv_tiles_per_head": tiles, "query_samples": qi.cpu().tolist(),
            "fp8_tiles": int(fp8_mask.sum().item()),
            "total_tiles": int(eligible.sum().item()),
            "fp8_tile_fraction": fp8_mask.float().mean().item(),
            "fp8_gemm_fraction": fp8_mask.float().mean().item(),
            "top1pct_fp8_tiles": int(budget_mask.sum().item()),
            "threshold_sweep_fp8_fraction": thresholds,
            "v_fp4_rel_l2_quantiles": {
                str(qt): torch.quantile(rel_l2[eligible], qt).item()
                for qt in (0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
            },
            "output_vs_bf16": metrics,
            "output_vs_qk_only": metrics_vs_qk_only,
            "bf16_output_energy": ref_energy,
        }
        return result, rel_l2.cpu().numpy(), fp8_mask.cpu().numpy(), budget_mask.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--main-queries", type=int, default=160)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must lie in (0,1)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    capture_sha = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    data = torch.load(args.capture, map_location="cpu", weights_only=True, mmap=True)
    assert data["format"] == "h3-480p3s-real-attention-v1"
    assert tuple(data["q"].shape) == (9536, 56, 128)
    bounds = data["cu_seqlens"].tolist()
    assert bounds == [0, 9482, 9536]
    scale = float(data["softmax_scale"])
    assert math.isclose(scale, 1 / math.sqrt(128), rel_tol=1e-9)
    output = {"format": "e112-h3-mixed-pv-reference-v1", "capture_sha256": capture_sha,
              "capture": str(args.capture), "block": data["block"],
              "prompt_id": data["prompt_id"], "seed": data["seed"],
              "threshold": args.threshold, "precision": "float32 PyTorch QDQ and GEMM",
              "fp8_rule": "E4M3FN, dynamic max/448 per 16 K elements for P and V",
              "caveat": "Output NMSE uses sampled main-sequence queries, not all query rows.",
              "sequences": {}}
    arrays = {}
    for name, start, end, nq in (("main", 0, 9482, args.main_queries),
                                  ("short", 9482, 9536, 54)):
        print(f"studying {name}: tokens={end-start}, queries={nq}", flush=True)
        result, errors, mask, budget = study_sequence(
            data["q"][start:end], data["k"][start:end], data["v"][start:end],
            scale, nq, args.threshold, device)
        output["sequences"][name] = result
        arrays[name] = (errors, mask, budget)
        print(json.dumps({"sequence": name, "fp8_tile_fraction": result["fp8_tile_fraction"],
                          "nmse": {k: v["nmse"] for k, v in result["output_vs_bf16"].items()}},
                         indent=2), flush=True)
        torch.cuda.empty_cache()

    main_result = output["sequences"]["main"]
    short_result = output["sequences"]["short"]
    total_energy = main_result["bf16_output_energy"] + short_result["bf16_output_energy"]
    output["complete_two_sequence_nmse"] = {
        arm: (
            main_result["bf16_output_energy"] * main_result["output_vs_bf16"][arm]["nmse"]
            + short_result["bf16_output_energy"] * short_result["output_vs_bf16"][arm]["nmse"]
        ) / total_energy
        for arm in main_result["output_vs_bf16"]
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tile_errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "head", "kv_tile", "v_fp4_rel_l2", "fp8_at_0p10", "fp8_top1pct"])
        for name, (errors, mask, budget) in arrays.items():
            for head in range(errors.shape[0]):
                for tile in range(errors.shape[1]):
                    writer.writerow([name, head, tile, float(errors[head, tile]),
                                     int(mask[head, tile]), int(budget[head, tile])])
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    errors, mask, budget = arrays["main"]
    fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True, layout="constrained")
    img = axs[0].imshow(errors, aspect="auto", cmap="viridis", origin="lower")
    fig.colorbar(img, ax=axs[0], label="V FP4 relative L2 error")
    axs[0].set_title("Real H3 block 25, prompt 1, step 0: V FP4 error")
    axs[1].imshow(mask, aspect="auto", cmap="Reds", vmin=0, vmax=1, origin="lower")
    axs[1].set_title(f"FP8 routing at threshold {args.threshold:.2f}: {mask.mean():.1%} of tiles/GEMMs")
    axs[2].imshow(budget, aspect="auto", cmap="Reds", vmin=0, vmax=1, origin="lower")
    axs[2].set_title("FP8 routing with a strict top-1% V-tile budget")
    for ax in axs:
        ax.set_ylabel("Head index")
    axs[-1].set_xlabel("KV tile index (128 tokens/tile)")
    fig.savefig(args.output_dir / "v_tile_error_and_fp8_route.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
