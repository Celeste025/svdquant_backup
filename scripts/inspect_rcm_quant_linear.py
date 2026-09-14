#!/usr/bin/env python3
"""Inspect how DeepCompressor represents selected rCM-Wan Linear modules."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from diffusers import WanPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/models/svdquant-wjq')
MODEL = DATA / 'models/rcm-Wan2.1-T2V-1.3B-Diffusers'
CKPT = DATA / 'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16'
NAMES = ('blocks.29.attn1.to_q', 'blocks.29.attn1.to_v', 'blocks.19.attn2.to_q')

p = ROOT / 'scripts/infer_rcm_wan_4step.py'
s = importlib.util.spec_from_file_location(p.stem, p); h = importlib.util.module_from_spec(s)
assert s.loader; s.loader.exec_module(h)
pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('cuda')
h.load_quantized_transformer(pipe, CKPT, MODEL)
for name in NAMES:
    m = pipe.transformer.get_submodule(name)
    print(f'[{name}]')
    print('type=', type(m), 'forward_hooks=', list(m._forward_hooks), 'pre_hooks=', list(m._forward_pre_hooks))
    print('children=', [(n, type(c).__module__ + '.' + type(c).__name__) for n, c in m.named_children()])
    print('attrs=', sorted(k for k in vars(m) if any(x in k.lower() for x in ('quant','scale','lowrank','branch','hook'))))
