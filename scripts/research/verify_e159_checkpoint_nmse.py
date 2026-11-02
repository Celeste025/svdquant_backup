#!/usr/bin/env python3
"""Independently recompute available E159 NMSEs from checkpoint tensors on CPU."""
import json
import math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research/E159'
torch.set_num_threads(4)
checks=[]
for pid in (25,48,44,7):
    ref=json.loads((ROOT/f'results/research/E150/p{pid:03d}/bf16_denoise.json').read_text())
    y=torch.load(ref['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].double()
    for mode in ('proxy','reuse'):
        p=OUT/f'aligned_v2/{mode}/p{pid:03d}/bf16_denoise.json'
        d=json.loads(p.read_text()) if p.exists() else {}
        if 'comparison' not in d:continue
        x=torch.load(d['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].double()
        assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        nmse=float((x-y).square().sum()/y.square().sum())
        recorded=d['comparison']['video_latent_nmse']
        assert math.isclose(nmse,recorded,rel_tol=1e-6,abs_tol=1e-9),(pid,mode,nmse,recorded)
        checks.append(dict(prompt_id=pid,mode=mode,shape=list(x.shape),recomputed_nmse=nmse,recorded_nmse=recorded,verified=True))
report={'status':'complete' if len(checks)==8 else 'partial','verified_arms':len(checks),'checks':checks}
(OUT/'checkpoint_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
