#!/usr/bin/env python3
"""Run VBench-2.0 dimension evaluation for one variant of the H3 100-prompt set.

Usage:
  python run_minimax_h3_vbench2_eval.py --variant bf16 --dimensions Camera_Motion Material
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VB2_ROOT = ROOT / "third_party/VBench/VBench-2.0"
METADATA = Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100")
CACHE = Path("/home/admin/workspace/aop_lab/app_data/cache/vbench2")

DIMENSION_GROUPS = {
    "llava": ["Human_Clothes", "Composition", "Dynamic_Spatial_Relationship", "Dynamic_Attribute",
              "Motion_Rationality", "Mechanics", "Thermotics", "Material"],
    "llava_qwen": ["Complex_Landscape", "Complex_Plot", "Human_Interaction", "Motion_Order_Understanding"],
    "tracking": ["Camera_Motion", "Multi-View_Consistency"],
    "identity": ["Human_Identity"],
    "anatomy": ["Human_Anatomy"],
    "instance": ["Instance_Preservation"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("bf16", "w4a4", "svdquant"), required=True)
    parser.add_argument("--dimensions", nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("VBENCH2_CACHE_DIR", str(CACHE))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(VB2_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(VB2_ROOT.parent))
    yolo_world_root = VB2_ROOT / "vbench2" / "third_party" / "YOLO-World"
    if str(yolo_world_root) not in sys.path:
        sys.path.insert(0, str(yolo_world_root))

    from vbench2 import VBench2  # noqa: E402  (imports after env vars are set)

    videos_path = args.metadata / "flat" / args.variant
    output_path = args.output or (args.metadata / "eval_results" / args.variant)
    full_info = args.metadata / "eval_full_info.json"

    my_vbench = VBench2(device=args.device, full_info_dir=str(full_info), output_path=str(output_path))
    for dim in args.dimensions:
        result_file = output_path / f"{args.variant}_{dim}_eval_results.json"
        if result_file.is_file():
            print(f"=== skip {args.variant} / {dim} (already evaluated) ===", flush=True)
            continue
        print(f"=== evaluating {args.variant} / {dim} ===", flush=True)
        my_vbench.evaluate(
            videos_path=str(videos_path),
            name=f"{args.variant}_{dim}",
            dimension_list=[dim],
            mode="vbench_standard",
        )


if __name__ == "__main__":
    main()
