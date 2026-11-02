#!/usr/bin/env python3
"""E146: p025 retained 20% as INT8 or equal INT8/NVFP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_e139_draftattention2 as worker
from anemoi import QuantConfig, SparseConfig

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E146"
DATA = Path("/data1/models/svdquant-wjq/research/20261101/E146_text_detail_retained20")
PLAN = REPORT / "experiment_plan.md"
PROMPTS = Path("/home/wjq/workspace/178866172854036")
CASE = "p025"
ARMS = ("s80_int8_20", "s80_nvfp4_10_int8_10")


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
        # Use the validated phase combinations, then set the E146 routing
        # budget and exact retained-precision ratio.
        base_arm = "s70_int8_30" if arm == "s80_int8_20" else "s70_fp4_20_int8_10"
        super().__init__(dit, base_arm)
        self.arm = arm
        self.sparse = SparseConfig(query_block_size=64, sparsity_ratio=0.80, maxpool_weight=0.2)
        self.quant = QuantConfig(
            nvfp4_ratio=0.0 if arm == "s80_int8_20" else 0.5,
            int8_ratio=1.0 if arm == "s80_int8_20" else 0.5,
            fp16_ratio=0.0,
            prefix_kv_precision="int8",
            prefix_query_precision="int8",
        )
        self.manifest.update(
            arm=arm,
            sparse_config=vars(self.sparse),
            quant_config=vars(self.quant),
            dense_first_steps=self.dense_first_steps,
            dense_first_layers=self.dense_first_layers,
            prefix_policy="official dense INT8",
        )


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
