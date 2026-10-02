#!/usr/bin/env python3
"""Aggregate VBench-2.0 eval results for the H3 100-prompt set into a score table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METADATA = Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100")
VARIANTS = ("bf16", "w4a4", "svdquant")

DIMENSION_CATEGORY = {
    "Human_Anatomy": "Human Fidelity", "Human_Identity": "Human Fidelity", "Human_Clothes": "Human Fidelity",
    "Human_Interaction": "Human Fidelity",
    "Camera_Motion": "Controllability",
    "Complex_Landscape": "Creativity", "Complex_Plot": "Creativity", "Composition": "Creativity",
    "Dynamic_Attribute": "Physics", "Dynamic_Spatial_Relationship": "Physics",
    "Motion_Order_Understanding": "Physics", "Motion_Rationality": "Physics",
    "Mechanics": "Physics", "Thermotics": "Physics", "Material": "Physics",
    "Instance_Preservation": "Controllability", "Multi-View_Consistency": "Controllability",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for variant in VARIANTS:
        result_dir = args.metadata / "eval_results" / variant
        scores = {}
        for path in sorted(result_dir.glob(f"{variant}_*_eval_results.json")):
            data = json.loads(path.read_text())
            for dim, payload in data.items():
                if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
                    scores[dim] = payload[0]
                elif isinstance(payload, dict) and "overall" in payload:
                    scores[dim] = payload["overall"]
                elif isinstance(payload, (int, float)):
                    scores[dim] = payload
        rows.append((variant, scores))

    dims = sorted({d for _, s in rows for d in s})
    header = f"{'dimension':32s} {'category':18s} " + " ".join(f"{v:>10s}" for v, _ in rows)
    lines = [header]
    for dim in dims:
        lines.append(f"{dim:32s} {DIMENSION_CATEGORY.get(dim, '-') :18s} "
                     + " ".join(f"{s.get(dim, float('nan')):10.4f}" for _, s in rows))
    table = "\n".join(lines)
    print(table)

    out = args.out or (args.metadata / "eval_results" / "score_table.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
