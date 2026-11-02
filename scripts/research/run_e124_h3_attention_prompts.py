#!/usr/bin/env python3
"""Run one E124 H3 prompt/attention arm using the validated E121 router."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

import run_e121_h3_split_generation as e121
from minimax_h3_svdquant_common import PROMPT_FILE, read_prompt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E124"
DATA = Path("/data1/models/svdquant-wjq/research/20261023/E124_h3_attention_12prompt")
MANIFEST = REPORT / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--arm", choices=e121.ARMS, required=True)
    parser.add_argument("--phase", choices=("denoise", "decode"), required=True)
    args = parser.parse_args()
    e121.check_backend()
    manifest = json.loads(MANIFEST.read_text())
    allowed = {int(x) for values in manifest["groups"].values() for x in values}
    e121.base.require(args.prompt_id in allowed and args.arm in manifest["arms"],
                      "prompt/arm is not in E124 manifest")
    e121.base.require(str(PROMPT_FILE) == manifest["prompt_source"],
                      "prompt source changed")
    e121.base.require(e121.base.sha256(PROMPT_FILE) == manifest["prompt_source_sha256"],
                      "prompt source hash changed")
    row = read_prompt(args.prompt_id)
    case = f"p{args.prompt_id:03d}"

    def selected_case_info(name: str) -> dict:
        e121.base.require(name == case, "case name changed")
        return {"prompt_id": args.prompt_id, "seed": int(row["seed"]),
                "prompt": row["prompt"]}

    e121.base.case_info = selected_case_info
    e121.base.REPORT = REPORT
    e121.base.DATA = DATA
    e121.base.PLAN = MANIFEST
    e121.base.Router = e121.Router
    importlib.import_module("diffsynth.core.attention.attention").ATTENTION_IMPLEMENTATION = "torch"
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    report = REPORT / case / f"{args.arm}_{'denoise' if args.phase == 'denoise' else 'video'}.json"
    if report.exists():
        existing = json.loads(report.read_text())
        complete = ((args.phase == "denoise" and existing.get("status") == "denoise_complete")
                    or (args.phase == "decode" and existing.get("frames") == 124))
        if complete:
            print(f"SKIP complete {case}/{args.arm}/{args.phase}", flush=True)
            return
        raise RuntimeError(f"Existing incomplete report requires inspection: {report}")

    if args.phase == "denoise":
        e121.base.denoise(case, args.arm)
    else:
        e121.base.decode(case, args.arm)


if __name__ == "__main__":
    main()
