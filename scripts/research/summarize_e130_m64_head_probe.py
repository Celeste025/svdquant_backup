#!/usr/bin/env python3
"""Summarize controlled M64 Sage3 outputs by head."""
from __future__ import annotations

import json
from pathlib import Path

import torch


FILES = {
    "ones_v_full56": Path("/tmp/e130_m64_candidate_ones_v_256.pt"),
    "ones_v_last28": Path("/tmp/e130_m64_candidate_ones_v_last28_256.pt"),
    "ones_v_last28_qk_0p1": Path("/tmp/e130_m64_candidate_ones_v_last28_qk01_256.pt"),
    "sentinel_ones_v_last28": Path("/tmp/e130_m64_sentinel_ones_v_last28_256.pt"),
    "stridepad_sentinel_ones_v_h4": Path("/tmp/e130_m64_stridepad_ones_v_h4_256.pt"),
}
OUTPUT = Path("results/research/E130/m64_controlled_head_summary.json")


def main() -> None:
    rows = {}
    for name, path in FILES.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        output = payload["output"]
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(f"unexpected output shape in {path}")
        zero_by_head = (output == 0).flatten(2).all(dim=-1)[0]
        sentinel_by_head = (output == 42).flatten(2).all(dim=-1)[0]
        rows[name] = {
            "source": str(path),
            "shape": list(output.shape),
            "head_start": payload.get("head_start", 0),
            "qk_scale": payload.get("qk_scale", 1.0),
            "all_zero_heads": zero_by_head.nonzero().flatten().tolist(),
            "all_sentinel_heads": sentinel_by_head.nonzero().flatten().tolist(),
            "head_means": output.float().mean(dim=(0, 2, 3)).tolist(),
        }
    if rows["ones_v_full56"]["all_zero_heads"] != list(range(28, 56)):
        raise RuntimeError("unexpected full-head zero pattern")
    if rows["ones_v_last28"]["all_zero_heads"] != list(range(14, 28)):
        raise RuntimeError("unexpected reindexed-head zero pattern")
    if rows["ones_v_last28_qk_0p1"]["all_zero_heads"] != list(range(14, 28)):
        raise RuntimeError("small QK did not reproduce zero pattern")
    if rows["sentinel_ones_v_last28"]["all_sentinel_heads"]:
        raise RuntimeError("sentinel remained in an entire head")
    report = {"format": "e130-m64-controlled-head-summary-v1", "cases": rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: value["all_zero_heads"] for name, value in rows.items()},
                     indent=2))


if __name__ == "__main__":
    main()
