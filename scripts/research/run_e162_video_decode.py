#!/usr/bin/env python3
"""Decode E162 proxy_new final latents using E150 VAE contract."""
import argparse
from pathlib import Path
import torch
import run_e150_whitebox_bf16 as base

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--pid', type=int, choices=(25, 48, 44, 7), required=True)
a = p.parse_args()
base.REPORT = ROOT / 'results/research/E162/proxy_new'
base.DATA = ROOT / 'results/local/E162/proxy_new'
base.ARM = 'proxy_new'
(base.DATA / f'p{a.pid:03d}' / 'proxy_new').mkdir(parents=True, exist_ok=True)
torch.set_num_threads(6)
base.decode(a.pid)
