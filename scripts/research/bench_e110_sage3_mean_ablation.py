#!/usr/bin/env python3
"""Sage3 FP4 Q/K centering and Q-mean correction ablation on captured H3 QKV.

All variants use the same native FP4 attention core.  The no-correction arms
pass a zero delta_s buffer, so their core still pays the delta_s load; online
timings measure the saved preprocessing work but are not a recompiled-kernel
upper bound.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from bench_h3_sage_real_gpu0 import metrics
from bench_h3_sage_two_scope import CAPTURE, measure


SAGE3 = Path("third_party/SageAttention3_prescale/sageattention3_blackwell")
OUTPUT = Path("results/research/E110/sage3_mean_ablation_h3_p1_s0_b25_gpu0.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sage3-dir", type=Path, default=SAGE3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--only-no-qk", action="store_true",
                        help="Run only the no-Q/K-mean arm and also time its prepacked core")
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir.resolve()))
    api = importlib.import_module("sageattn3_prescale.api")
    torch.cuda.set_device(args.gpu)

    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    assert payload["format"] == "h3-480p3s-real-attention-v1"
    n, heads, dim = 9482, 56, 128
    device = f"cuda:{args.gpu}"
    q, k, v = [payload[name][:n].to(device).transpose(0, 1).unsqueeze(0)
               for name in ("q", "k", "v")]
    assert q.shape == k.shape == v.shape == (1, heads, n, dim)
    k_work = torch.empty_like(k)
    padded_n = ((n + 127) // 128) * 128
    zero_delta = torch.zeros((1, heads, padded_n // 128, padded_n),
                             dtype=torch.float32, device=device)
    operations = 4 * heads * n * n * dim
    arms = {
        "baseline": (True, True, True),
        "no_q_mean_no_correction": (False, True, False),
        "no_k_mean": (True, False, True),
        "no_qk_means_no_correction": (False, False, False),
    }
    if args.only_no_qk:
        arms = {"no_qk_means_no_correction": arms["no_qk_means_no_correction"]}

    def pad_128(x: torch.Tensor) -> torch.Tensor:
        if n == padded_n:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, padded_n - n)).contiguous()

    def make_online(center_q: bool, center_k: bool, correction: bool):
        def online():
            kk = k_work
            if center_k:
                kk -= kk.mean(dim=-2, keepdim=True)
            q3, k3, v3 = [pad_128(x) for x in (q, kk, v)]
            if center_q:
                q3, qm = api.triton_group_mean(q3)
            if correction:
                delta_s = torch.matmul(qm, k3.transpose(-2, -1)).to(torch.float32).contiguous()
            else:
                delta_s = zero_delta
            qlist = api.scale_and_quant_fp4(q3)
            klist = api.scale_and_quant_fp4_permute(k3)
            vlist = api.scale_and_quant_fp4_transpose(v3)
            return api.blockscaled_fp4_attn(qlist, klist, vlist, delta_s, n,
                                            is_causal=False, per_block_mean=True,
                                            is_bf16=True)[0][:, :, :n, :].contiguous()
        return online

    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, scale=dim ** -0.5)
        results = {}
        for name, switches in arms.items():
            fn = make_online(*switches)
            prepare = lambda: k_work.copy_(k)
            prepare()
            out = fn()
            err = metrics(out, ref)
            results[name] = {
                "center_q": switches[0], "center_k": switches[1],
                "q_mean_correction": switches[2],
                "nmse_vs_bf16": err["nmse"],
                "cosine_vs_bf16": err.get("cosine"),
            }
            print(f"{name}: NMSE={err['nmse']:.9f}", flush=True)
            # Repeat in separate rounds to expose drift without suppressing it.
            results[name]["rounds"] = []

        agreement = None
        if not args.only_no_qk:
            # Confirm our baseline reconstructs the public API exactly.
            k_work.copy_(k)
            public = api.sageattn3_blackwell(q, k_work, v, is_causal=False,
                                             per_block_mean=True)
            k_work.copy_(k)
            custom = make_online(*arms["baseline"])()
            agreement = metrics(custom, public)["nmse"]
            if agreement > 1e-8:
                raise RuntimeError(f"Custom baseline differs from public API: {agreement}")

        if args.only_no_qk:
            q3, k3, v3 = [pad_128(x) for x in (q, k, v)]
            qlist = api.scale_and_quant_fp4(q3)
            klist = api.scale_and_quant_fp4_permute(k3)
            vlist = api.scale_and_quant_fp4_transpose(v3)

            def core():
                return api.blockscaled_fp4_attn(qlist, klist, vlist, zero_delta, n,
                                                is_causal=False, per_block_mean=True,
                                                is_bf16=True)[0]

            core_error = metrics(core()[:, :, :n, :].contiguous(), ref)["nmse"]
            if abs(core_error - results["no_qk_means_no_correction"]["nmse_vs_bf16"]) > 1e-7:
                raise RuntimeError("Prepacked core output disagrees with online no-Q/K-mean arm")
            core_rounds = [measure(core) for _ in range(2)]
            results["no_qk_means_no_correction"]["core_mean_ms"] = (
                sum(t["mean_ms"] for t in core_rounds) / len(core_rounds))
            results["no_qk_means_no_correction"]["core_rounds"] = core_rounds
            print(f"no_qk_means_no_correction core: "
                  f"{results['no_qk_means_no_correction']['core_mean_ms']:.4f} ms", flush=True)

        for order in (tuple(arms), tuple(reversed(arms))):
            for name in order:
                fn = make_online(*arms[name])
                timing = measure(fn, lambda: k_work.copy_(k))
                results[name]["rounds"].append(timing)
                print(f"{name}: {timing['mean_ms']:.4f} ms; "
                      f"CV={timing['cv_batch_mean_percent']:.2f}%", flush=True)

    for row in results.values():
        row["mean_ms"] = sum(t["mean_ms"] for t in row["rounds"]) / len(row["rounds"])
        row["tops"] = operations / 1e12 / (row["mean_ms"] / 1000)
    report = {
        "format": "e110-sage3-fp4-mean-ablation-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.sage3_dir.resolve()), "capture": str(CAPTURE),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "shape_hnd": [1, heads, n, dim], "is_causal": False,
        "baseline_custom_vs_public_nmse": agreement,
        "only_no_qk": args.only_no_qk,
        "timing": "CUDA events, 25 warmups + 12x12 calls per round; 2 reversed-order rounds; K reset excluded",
        "caveat": "No-correction arms still pass a preallocated zero delta_s to the unchanged native core.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
