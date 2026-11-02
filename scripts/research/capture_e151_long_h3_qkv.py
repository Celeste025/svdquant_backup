#!/usr/bin/env python3
"""Capture real H3 block-0 QKV at first BF16 denoising call."""
import argparse
import importlib
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy

DATA=Path('/data1/models/svdquant-wjq/research/20261101/E151_long_h3_attention')
class Captured(Exception):pass

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frames',type=int,choices=(124,362),required=True);a=ap.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
    row=read_prompt(16);pipe=load_h3_pipeline(full=True,vram_limit_gib=18.0);pipe.load_models_to_device(['dit']);pipe.dit.eval()
    grid=(((a.frames-5)//17)*5+2,24,42);active=[False];layout={};packet={}
    def bind(module,args,kwargs):
        ids=kwargs['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long();start=int(ids[0]);assert ids.numel()==grid[0]*grid[1]*grid[2];assert torch.equal(ids,torch.arange(start,start+ids.numel()));layout['prefix']=start
    h0=pipe.dit.register_forward_pre_hook(bind,with_kwargs=True)
    h1=pipe.dit.blocks[0].attn.register_forward_pre_hook(lambda *args:active.__setitem__(0,True))
    original=comfy._sdpa_varlen_attention
    def intercept(q,k,v,cu_seqlens,softmax_scale):
        if not active[0]:return original(q,k,v,cu_seqlens,softmax_scale)
        bounds=cu_seqlens.detach().cpu().tolist();n=bounds[1]
        assert n==layout['prefix']+grid[0]*grid[1]*grid[2] and q.shape==k.shape==v.shape and q.dtype==torch.bfloat16
        for name,x in zip(('q','k','v'),(q,k,v)):packet[name]=x[:n].transpose(0,1).unsqueeze(0).cpu().contiguous()
        packet.update(prefix_tokens=layout['prefix'],video_grid=grid,layer=0,step=0,case='race',frames=a.frames,bounds=bounds,softmax_scale=float(softmax_scale))
        raise Captured()
    comfy._sdpa_varlen_attention=intercept
    try:
        try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=768,width=1344,num_frames=a.frames,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
        except Captured:pass
        else:raise RuntimeError('capture not reached')
    finally:h0.remove();h1.remove();comfy._sdpa_varlen_attention=original
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/f'race_768p_{a.frames}f_step0_block0.pt'
    with path.open('xb') as f:torch.save(packet,f)
    print(f'CAPTURED {path} q={tuple(packet["q"].shape)} grid={grid} prefix={layout["prefix"]}',flush=True)

if __name__=='__main__':main()
