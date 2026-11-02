#!/usr/bin/env python3
import argparse, hashlib, importlib.util, inspect, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--input-size',choices=('480','768'),required=True)
a,remaining=ap.parse_known_args();sys.argv=[sys.argv[0],*remaining]
name='validate_e169_480p_paths.py' if a.input_size=='480' else 'validate_e163_four_paths.py'
spec=importlib.util.spec_from_file_location('validation',ROOT/'scripts/research'/name)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
checks={}
def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
@torch.inference_mode()
def measure(function,**kwargs):
    output=function();checks[function.__name__]={'output':digest(output)}
    cache=inspect.getclosurevars(function).nonlocals.get('cache')
    if cache is not None:
        plan=next(iter(cache._routes.values())).plan
        checks[function.__name__]['route']={field:digest(getattr(plan,field)) for field in
          ('block_ids','packed_offsets','nvfp4_counts','fp8_counts','fp16_counts')}
        checks[function.__name__]['activity']={}
        for field in ('nvfp4_counts','fp8_counts','fp16_counts'):
            counts=getattr(plan,field).cpu()
            checks[function.__name__]['activity'][field]={
              'rows':counts.numel(),'inactive_rows':int((counts==0).sum()),
              'min':int(counts.min()),'max':int(counts.max()),
              'mean':float(counts.float().mean())}
    return {'mean_ms':0,'scope':'output hash only; not a timing measurement'}
v._measure=measure
v.main()
target=Path(sys.argv[sys.argv.index('--output')+1])
target.with_name(target.stem+'_hashes.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
