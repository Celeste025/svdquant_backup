#!/usr/bin/env python3
"""Evaluate E148 against BF16 and the 80/0/20 INT8 baseline."""
from pathlib import Path
import finish_e147_paper_pair as base

root = Path(__file__).resolve().parents[2]
base.OUT = root / 'results/research/E148/p025'
base.DATA = Path('/data1/models/svdquant-wjq/research/20261101/E148_mix15_10/p025')
base.records = [
    (root / 'results/research/E124/p025', 'bf16'),
    (root / 'results/research/E146/p025', 's80_int8_20'),
    (base.OUT, 's75_fp415_int810'),
]

if __name__ == '__main__':
    base.main()
