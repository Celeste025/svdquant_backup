#!/usr/bin/env python3
"""E145: p025 retained 30% pure INT8 versus pure NVFP4 attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_e139_draftattention2 as worker
from anemoi import QuantConfig

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E145"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E145_text_detail_pure_precision")
PLAN = REPORT / "experiment_plan.md"
PROMPTS = Path("/home/wjq/workspace/178866172854036")
CASE = "p025"
ARMS = ("s70_int8_30", "s70_nvfp4_30")


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

    def __init__(self, dit, arm: str):
        if arm == "s70_nvfp4_30":
            # Initialize the validated mixed native path, then make every
            # retained video block NVFP4.  With no active video INT8 phase the
            # native API requires prefix-query BF16; prefix K/V remain INT8.
            super().__init__(dit, "s70_fp4_20_int8_10")
            self.arm = arm
            self.quant = QuantConfig(
                nvfp4_ratio=1.0,
                int8_ratio=0.0,
                fp16_ratio=0.0,
                prefix_kv_precision="int8",
                prefix_query_precision="fp16",
            )
            self.manifest.update(
                arm=arm,
                quant_config=vars(self.quant),
                prefix_policy="K/V INT8; query BF16 dense because video INT8 phase is absent",
            )
        else:
            super().__init__(dit, arm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
        worker.e121.base.denoise(CASE, args.arm)
    else:
        worker.e121.base.decode(CASE, args.arm)


if __name__ == "__main__":
    main()
