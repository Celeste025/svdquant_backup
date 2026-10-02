#!/usr/bin/env python3
"""Populate the torch.hub cache for facebookresearch/co-tracker so that
`torch.hub.load('facebookresearch/co-tracker', 'cotracker2')` works without
GitHub access (github.com is unreachable from this host; torch.hub falls back
to `<owner>_<repo>_<branch>` cache dirs when the network fails)."""
from __future__ import annotations

import shutil
from pathlib import Path

HUB = Path("/home/admin/workspace/aop_lab/app_data/cache/vbench2/cotracker_hub")
REPO_DIR = HUB
VENDORED = Path("/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup/third_party/VBench/VBench-2.0/vbench2/third_party/cotracker")
WEIGHT = Path("/home/admin/workspace/aop_lab/app_data/cache/vbench2/cotracker_hub/cotracker2.pth")

HUBCONF = '''dependencies = ["torch"]

import os

from cotracker.predictor import CoTrackerPredictor

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_ROOT, "checkpoints", "cotracker2.pth")


def cotracker2(pretrained=True, window_len=8):
    if not pretrained:
        return CoTrackerPredictor(checkpoint=None, v2=True, window_len=window_len)
    return CoTrackerPredictor(checkpoint=_CKPT, v2=True, window_len=window_len)
'''


def main() -> None:
    pkg = REPO_DIR / "cotracker"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)
    for item in ("models", "utils", "predictor.py", "__init__.py", "version.py"):
        src = VENDORED / item
        if src.is_dir():
            shutil.copytree(src, pkg / item)
        else:
            shutil.copy2(src, pkg / item)
    (REPO_DIR / "checkpoints").mkdir(exist_ok=True)
    ckpt = REPO_DIR / "checkpoints" / "cotracker2.pth"
    if not ckpt.exists():
        shutil.copy2(WEIGHT, ckpt)
    (REPO_DIR / "hubconf.py").write_text(HUBCONF)
    print(f"populated {REPO_DIR}")
    for p in sorted(REPO_DIR.rglob("*")):
        if p.is_dir() and "__pycache__" in str(p):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
