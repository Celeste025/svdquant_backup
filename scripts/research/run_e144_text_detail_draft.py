#!/usr/bin/env python3
"""E144: p025 text/detail case with first 2 steps + block 0 dense."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_e139_draftattention2 as worker

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E144"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E144_text_detail_draft")
PLAN = REPORT / "experiment_plan.md"
PROMPTS = Path("/home/wjq/workspace/178866172854036")
CASE = "p025"
ARM = "s70_fp4_20_int8_10"


def prompt25() -> dict:
    with PROMPTS.open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["prompt_id"]) == 25:
                return {"prompt_id": 25, "seed": int(row["seed"]), "prompt": row["prompt"]}
    raise RuntimeError("prompt_id 25 missing")


class Router(worker.Router):
    dense_first_steps = 2
    dense_first_layers = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("denoise", "decode"), required=True)
    args = parser.parse_args()
    row = prompt25()
    worker.e121.base.require(row["seed"] == 2141, "p025 seed changed")
    worker.e121.base.case_info = lambda case: row if case == CASE else (_ for _ in ()).throw(
        RuntimeError(f"unexpected case {case}")
    )
    worker.e121.base.REPORT = REPORT
    worker.e121.base.DATA = DATA
    worker.e121.base.PLAN = PLAN
    worker.e121.base.Router = Router
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.phase == "denoise":
        worker.e121.base.denoise(CASE, ARM)
    else:
        worker.e121.base.decode(CASE, ARM)


if __name__ == "__main__":
    main()
