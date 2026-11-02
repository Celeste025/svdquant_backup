#!/usr/bin/env python3
"""Run the matched E125 clap0 BF16 / corrected Sage3 / split mixed arms."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

import run_e121_h3_split_generation as e121


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E125"
DATA = Path("/data1/models/svdquant-wjq/research/20261023/E125_h3_clap_attention")
MANIFEST = REPORT / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=e121.ARMS, required=True)
    parser.add_argument("--phase", choices=("denoise", "decode"), required=True)
    args = parser.parse_args()
    e121.check_backend()
    manifest = json.loads(MANIFEST.read_text())
    info = e121.base.case_info("clap0")
    e121.base.require(
        manifest["format"] == "e125-h3-clap-attention-triplet-v1" and
        (info["prompt_id"], info["prompt"], info["seed"]) ==
        (manifest["prompt_id"], manifest["prompt"], manifest["seed"]) and
        args.arm in manifest["arms"], "E125 clap identity changed")
    e121.base.REPORT = REPORT
    e121.base.DATA = DATA
    e121.base.PLAN = MANIFEST
    e121.base.Router = e121.Router
    importlib.import_module("diffsynth.core.attention.attention").ATTENTION_IMPLEMENTATION = "torch"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    report = REPORT / "clap0" / f"{args.arm}_{'denoise' if args.phase == 'denoise' else 'video'}.json"
    if report.exists():
        existing = json.loads(report.read_text())
        complete = ((args.phase == "denoise" and existing.get("status") == "denoise_complete")
                    or (args.phase == "decode" and existing.get("frames") == 124))
        if complete:
            print(f"SKIP complete clap0/{args.arm}/{args.phase}", flush=True)
            return
        raise RuntimeError(f"Existing incomplete report requires inspection: {report}")
    if args.phase == "denoise":
        e121.base.denoise("clap0", args.arm)
    else:
        e121.base.decode("clap0", args.arm)


if __name__ == "__main__":
    main()
