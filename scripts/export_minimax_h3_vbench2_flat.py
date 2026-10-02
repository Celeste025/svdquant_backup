#!/usr/bin/env python3
"""Export generated H3 vbench2 videos into the flat layout VBench expects:
<variant>/<prompt[:180]>-0.mp4 (symlinks into cases/<case_id>/<variant>.mp4)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFEST = Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100/manifest.json")
VARIANTS = ("bf16", "w4a4", "svdquant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cases-root", type=Path, default=MANIFEST.parent / "cases")
    parser.add_argument("--out-root", type=Path, default=MANIFEST.parent / "flat")
    args = parser.parse_args()

    cases = json.loads(args.manifest.read_text())["cases"]
    for variant in VARIANTS:
        out_dir = args.out_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        linked = missing = 0
        for case in cases:
            src = args.cases_root / case["case_id"] / f"{variant}.mp4"
            dst = out_dir / f"{case['prompt'][:180]}-0.mp4"
            if not src.is_file():
                missing += 1
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            linked += 1
        print(f"{variant}: linked {linked}, missing {missing} -> {out_dir}")


if __name__ == "__main__":
    main()
