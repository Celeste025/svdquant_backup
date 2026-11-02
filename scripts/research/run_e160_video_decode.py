#!/usr/bin/env python3
"""Decode E159 final latents using the E150 VAE contract to separate outputs."""
import argparse
from pathlib import Path
import torch
import run_e150_whitebox_bf16 as base
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--mode',choices=('proxy','reuse'),required=True);a=p.parse_args()
base.REPORT=ROOT/'results/research/E159/aligned_v2'/a.mode
base.DATA=ROOT/'results/local/E160'/a.mode
base.ARM=a.mode
(base.DATA/f'p{a.pid:03d}'/a.mode).mkdir(parents=True,exist_ok=True)
torch.set_num_threads(6)
base.decode(a.pid)
