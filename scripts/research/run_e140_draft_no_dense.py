#!/usr/bin/env python3
"""E140: official DraftAttention2 on all 50 H3 main blocks and all 20 steps."""
import argparse
from pathlib import Path
import torch
import run_e139_draftattention2 as driver

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'results/research/E140'
DATA = Path('/data1/models/svdquant-wjq/research/20261031/E140_draft_no_dense')


class Router(driver.Router):
    dense_first_steps = 0
    dense_first_layers = 0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',choices=['race','clap0'],required=True)
    ap.add_argument('--arm',choices=['s70_int8_30','s70_fp4_20_int8_10'],required=True)
    ap.add_argument('--phase',choices=['denoise','decode'],required=True)
    args=ap.parse_args()
    base=driver.e121.base
    base.REPORT,base.DATA=REPORT,DATA
    base.PLAN=REPORT/'experiment_plan.md'
    base.Router=Router
    torch.set_num_threads(6)
    if args.phase=='denoise':
        base.denoise(args.case,args.arm)
    else:
        base.decode(args.case,args.arm)


if __name__=='__main__':
    main()
