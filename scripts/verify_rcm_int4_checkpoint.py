#!/usr/bin/env python3
"""Validate and describe the rCM INT4 checkpoint after PTQ completes."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from diffusers import WanPipeline, WanTransformer3DModel

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--model',type=Path,required=True); p.add_argument('--transformer',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); a=p.parse_args()
    import sys; sys.path.insert(0,str(ROOT/'scripts'))
    from infer_rcm_wan_4step import load_quantized_transformer
    pipe=WanPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16); pipe.transformer=WanTransformer3DModel.from_pretrained(a.transformer,torch_dtype=torch.bfloat16); pipe=pipe.to('cuda')
    load_quantized_transformer(pipe,a.checkpoint,a.model)
    linears=[m for m in pipe.transformer.modules() if isinstance(m,torch.nn.Linear)]
    if len(linears)!=300: raise RuntimeError(f'expected 300 Linear modules, got {len(linears)}')
    # INT4 checkpoints have no FP4 two-level scale entries.  Dynamic input
    # quantizers are installed as module pre-hooks by DeepCompressor.
    state=torch.load(a.checkpoint/'wgts.pt',map_location='cpu',weights_only=False)
    text=repr(state)
    if 'sfp4' in text.lower() or 'e4m3' in text.lower(): raise RuntimeError('checkpoint contains NVFP4 state')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    args=cache['input_args']; kwargs=cache['input_kwargs']
    def to_cuda(v):
        if torch.is_tensor(v): return v.to('cuda')
        if isinstance(v,list): return [to_cuda(x) for x in v]
        if isinstance(v,tuple): return tuple(to_cuda(x) for x in v)
        if isinstance(v,dict): return {k:to_cuda(x) for k,x in v.items()}
        return v
    with torch.inference_mode(): out=pipe.transformer(*to_cuda(args), **to_cuda(kwargs))[0]
    if not torch.isfinite(out).all(): raise RuntimeError('non-finite cached forward')
    manifest={'format':'rcm-wan-int4-svdquant-v1','quantization':{'weights':'dynamic-range signed INT4, group64','activations':'dynamic INT4, group64, unsigned permitted for nonnegative groups','activation_shift':True},'svdquant':{'rank':32,'smooth_grids':10,'lowrank_iters':100},'calibration':{'cache_root':str(a.cache.parent.parent),'samples':64,'prompts':16,'rcm_steps':4,'frames':77,'height':480,'width':832,'sigma_max':80.0},'model':str(a.model),'transformer':str(a.transformer),'validation':{'linears':len(linears),'cached_forward_finite':True}}
    (a.checkpoint/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
