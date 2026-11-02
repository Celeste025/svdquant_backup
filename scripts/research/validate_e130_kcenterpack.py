#!/usr/bin/env python3
"""Validate fused K-centering/NVFP4 packing against Sage3's original path."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/research"),
                str(ROOT / "third_party/sage3_h3_maskfix")]

from bench_h3_sage_real_gpu0 import metrics
from e130_sage3_preprocess import prepack

CAPTURES = (
    (Path("/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt"), 9482),
    (Path("/data1/models/svdquant-wjq/research/20261023/E121_h3_split_generation/race/block25_qkv_h56.pt"), 22677),
)
OUTPUT = ROOT / "results/research/E130/kcenterpack_validation_gpu2.json"


def main() -> None:
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as api

    torch.cuda.set_device(0)
    cases = []
    with torch.inference_mode():
        for capture, n in CAPTURES:
            payload = torch.load(capture, map_location="cpu", weights_only=True, mmap=True)
            if payload["q"].ndim == 3:
                q, k, v = [payload[name][:n].to("cuda:0").transpose(0, 1).unsqueeze(0)
                           for name in ("q", "k", "v")]
            else:
                q, k, v = [payload[name].to("cuda:0") for name in ("q", "k", "v")]
            reference = sageattn3_blackwell(q, k.clone(), v,
                                            is_causal=False, per_block_mean=True)
            qo, ko, vo, dso = api.preprocess_qkv(q, k.clone(), v, True)
            ko_pack = api.scale_and_quant_fp4_permute(ko)
            k_work = k.clone()
            qpack, kpack, vpack, ds, _ = prepack(
                q, k_work, v, api, fuse_q_pad_mean=True,
                implicit_kv_pad=True, fuse_k_center_pack=True,
            )
            candidate = api.blockscaled_fp4_attn(qpack, kpack, vpack, ds, n,
                                                  is_causal=False, per_block_mean=True,
                                                  is_bf16=True)[0][:, :, :n, :].contiguous()
            comparison = metrics(candidate, reference)
            row = {
                "tokens": n, "capture": str(capture),
                "k_center_equal": bool(torch.equal(k_work, ko[:, :, :n, :])),
                "k_center_mismatch_count": int((k_work != ko[:, :, :n, :]).sum()),
                "k_code_equal": bool(torch.equal(kpack[0], ko_pack[0])),
                "k_scale_equal": bool(torch.equal(kpack[1], ko_pack[1])),
                "delta_max_abs_difference": float((ds - dso).abs().max()),
                "output_vs_official": comparison,
                "finite": bool(torch.isfinite(candidate).all()),
            }
            row["passed"] = (row["k_center_equal"] and row["k_code_equal"]
                             and row["k_scale_equal"] and row["finite"]
                             and comparison["nmse"] <= 1e-7)
            print(json.dumps(row), flush=True)
            cases.append(row)
            del q, k, v, reference, qo, ko, vo, dso, ko_pack, k_work
    result = {"format": "e130-k-center-pack-validation-v1", "cases": cases,
              "passed": all(row["passed"] for row in cases)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("K center/pack fusion is not equivalent")


if __name__ == "__main__":
    main()
