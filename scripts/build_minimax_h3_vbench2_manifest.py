#!/usr/bin/env python3
"""Build the deterministic 100-prompt VBench-2.0 subset manifest for MiniMax-H3."""
from __future__ import annotations

import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_INFO = ROOT / "third_party/VBench/VBench-2.0/vbench2/VBench2_full_info.json"
OUT = Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100")
SETTINGS = {"seed": 0, "height": 576, "width": 1024, "frames": 124, "steps": 20,
            "cfg_scale": 1.0, "rand_device": "cpu", "tiled": True, "fps": 24}
# 100 prompts across 17 dimensions (same proportional ordering as the 54-case
# pilot, scaled ~1.9x); Diversity excluded (needs 20 videos per prompt).
QUOTAS = OrderedDict([
    ("Human_Anatomy", 9), ("Camera_Motion", 9), ("Human_Interaction", 8),
    ("Multi-View_Consistency", 8), ("Dynamic_Attribute", 8), ("Motion_Order_Understanding", 6),
    ("Human_Clothes", 6), ("Dynamic_Spatial_Relationship", 6), ("Complex_Plot", 6),
    ("Motion_Rationality", 5), ("Instance_Preservation", 5), ("Mechanics", 5),
    ("Composition", 4), ("Human_Identity", 4), ("Material", 4), ("Thermotics", 4),
    ("Complex_Landscape", 3),
])
TOTAL = sum(QUOTAS.values())
STATE = Path("/home/admin/workspace/aop_lab/app_data/artifacts/variants/nvfp4-g10-r64/quant_state.pt")
EVAL_DIMS = set(QUOTAS)


def equidistant(n: int, k: int) -> list[int]:
    if k == 1:
        return [n // 2]
    return sorted({i * (n - 1) // (k - 1) for i in range(k)})


def main() -> None:
    full_info = json.loads(FULL_INFO.read_text())
    pool: dict[str, list[str]] = defaultdict(list)
    entries: dict[tuple[str, str], dict] = {}
    for entry in full_info:
        for dim in entry["dimension"]:
            if entry["prompt_en"] not in pool[dim]:
                pool[dim].append(entry["prompt_en"])
            entries[(entry["prompt_en"], dim)] = entry
    for dim in pool:
        pool[dim].sort()
        if len(pool[dim]) < QUOTAS.get(dim, 0):
            raise RuntimeError(f"{dim}: pool {len(pool[dim])} < quota {QUOTAS[dim]}")

    selected: dict[str, list[str]] = {}
    counts = {}
    for dim, quota in QUOTAS.items():
        taken = set(selected)
        picked = []
        for anchor in equidistant(len(pool[dim]), quota):
            idx = anchor
            while idx < len(pool[dim]) and pool[dim][idx] in taken:
                idx += 1
            if idx >= len(pool[dim]):  # pool edge reached: fall back to nearest untaken
                idx = anchor
                while idx >= 0 and pool[dim][idx] in taken:
                    idx -= 1
            if idx < 0:
                raise RuntimeError(f"{dim}: exhausted pool after anchor {anchor}")
            picked.append(pool[dim][idx])
            taken.add(pool[dim][idx])
        counts[dim] = len(picked)
        for caption in picked:
            selected.setdefault(caption, []).append(dim)

    if sum(counts.values()) != TOTAL or counts != dict(QUOTAS):
        raise RuntimeError(f"quota mismatch: {counts}")
    cases = []
    for i, (caption, dims) in enumerate(sorted(selected.items())):
        if "/" in caption or caption[-2:] in (f"-{d}" for d in range(10)):
            raise RuntimeError(f"unsafe prompt for filename: {caption!r}")
        aux = {dim: entries[(caption, dim)]["auxiliary_info"]
               for dim in dims if entries[(caption, dim)].get("auxiliary_info")}
        cases.append({"case_id": f"vbench2_{i:03d}", "prompt": caption,
                      "dimensions": dims, "auxiliary_info": aux})
    keys = [c["prompt"][:180] for c in cases]
    if len(set(keys)) != len(keys):
        raise RuntimeError("prompt[:180] collision")

    eval_entries = [{"prompt_en": case["prompt"], "dimension": [dim],
                     **({"auxiliary_info": case["auxiliary_info"][dim]}
                        if case["auxiliary_info"].get(dim) else {})}
                    for case in cases for dim in case["dimensions"]]

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "source": str(FULL_INFO),
        "created_at": time.time(), "settings": SETTINGS,
        "sampling": {"method": "deterministic equidistant per dimension",
                     "source_population": len(full_info), "cases": len(cases),
                     "dimension_quotas": dict(QUOTAS), "dimension_counts": counts,
                     "notes": "Diversity excluded (requires 20 videos/prompt); convrot variant dropped (state unavailable)"},
        "variants": {"bf16": {"format": "bf16"},
                     "w4a4": {"format": "real-NVFP4 dynamic W4A4", "smoothing": False, "low_rank": False,
                              "group_size": 16, "activation_element_size": 128},
                     "svdquant": {"format": "real-NVFP4 SVDQuant", "rank": 64, "grid": 10,
                                  "max_lowrank_iters": 50, "state": str(STATE)}},
        "cases": cases,
    }
    for path, payload in ((OUT / "manifest.json", manifest), (OUT / "eval_full_info.json", eval_entries)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)
    print(f"wrote {OUT / 'manifest.json'} ({len(cases)} cases) and eval_full_info.json ({len(eval_entries)} entries)")
    print("multi-dimension cases:", sum(1 for c in cases if len(c["dimensions"]) > 1))
    for case in cases[:3]:
        print(" ", case["case_id"], case["dimensions"], case["prompt"][:60])


if __name__ == "__main__":
    main()
