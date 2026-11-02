#!/usr/bin/env python3
"""E141: first2 steps dense, then all50 main blocks sparse for18 steps."""
from pathlib import Path
import run_e140_draft_no_dense as worker

ROOT=Path(__file__).resolve().parents[2]


class Router(worker.Router):
    dense_first_steps=2
    dense_first_layers=0


if __name__=='__main__':
    worker.REPORT=ROOT/'results/research/E141'
    worker.DATA=Path('/data1/models/svdquant-wjq/research/20261031/E141_draft_first2_dense')
    worker.Router=Router
    worker.main()
