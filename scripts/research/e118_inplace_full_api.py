#!/usr/bin/env python3
"""Full online API timing of the in-place FP8 accumulator candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = Path("/data1/models/svdquant-wjq/research/20261014/E098/"
               "h3_p1_s0_b25_480x832_73f_qkv.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/research/E118/full_api_gpu0.json")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "third_party/sage3_h3_maskfix"),
                    str(ROOT / "third_party/sage3_native_mixed"),
                    str(ROOT / "scripts/research")]
    import fp4attn_mixed_maskfix_cuda
    import fp4attn_mixed_inplace_cuda
    import sage3_mixed_router_sparse_cuda as sparse_router
    from e117_full_api_benchmark import time_cuda
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (preprocess_qkv, scale_and_quant_fp4,
                               scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose)

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (payload[key][:9482].to(f"cuda:{args.gpu}")
               .transpose(0, 1).unsqueeze(0) for key in ("q", "k", "v"))
    k_work = torch.empty_like(k)

    def prepare():
        k_work.copy_(k)

    def mixed(core):
        qp, kp, vp, ds = preprocess_qkv(q, k_work, v, True)
        q4 = scale_and_quant_fp4(qp)
        k4 = scale_and_quant_fp4_permute(kp)
        v4 = scale_and_quant_fp4_transpose(vp)
        v8, scale8, route, _ = sparse_router.route_and_pack_v(vp, 0.10)
        packet = (q4[0], k4[0], v4[0], q4[1], k4[1], v4[1],
                  ds, 9482, None, 128 ** -0.5, False, True, True)
        return core.fwd_mixed(*packet, v8, scale8, route)[0][:, :, :9482]

    with torch.inference_mode():
        prepare()
        base = mixed(fp4attn_mixed_maskfix_cuda)
        prepare()
        candidate = mixed(fp4attn_mixed_inplace_cuda)
        error = float((base.float() - candidate.float()).square().sum() /
                      base.float().square().sum())
        assert error < 1e-9
        report = {
            "format": "e118-inplace-full-api-v1",
            "gpu": torch.cuda.get_device_name(args.gpu),
            "capture": str(CAPTURE), "shape": list(q.shape),
            "timing_excludes_benchmark_only_k_restore": True,
            "candidate_vs_baseline_nmse": error,
            "timing": {
                "pure_fixed_online": time_cuda(lambda:
                    sageattn3_blackwell(q, k_work, v, is_causal=False,
                                       per_block_mean=True), prepare),
                "mixed_sparse_baseline_online": time_cuda(lambda:
                    mixed(fp4attn_mixed_maskfix_cuda), prepare),
                "mixed_sparse_inplace_online": time_cuda(lambda:
                    mixed(fp4attn_mixed_inplace_cuda), prepare),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_vs_baseline_nmse": error,
                      "timing_ms": {k: v["mean_ms"] for k, v in
                                    report["timing"].items()}}, indent=2))


if __name__ == "__main__":
    main()
