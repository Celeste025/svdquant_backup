#!/usr/bin/env python3
"""Capture three true post-normalization/RoPE H3 QKV calls from BF16 caches."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research')]
from minimax_h3_svdquant_common import load_h3_pipeline, tree_device
from diffsynth.models import minimax_h3_dit_comfy as comfy

SPEC={'race_s00_b00':('race',0,0),'race_s10_b25':('race',10,25),'clap_s10_b49':('clap0',10,49)}
DATA=Path('/data1/models/svdquant-wjq/research/20261031/E142_attention_tradeoff')


class Captured(Exception):
    pass


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample',choices=SPEC,required=True)
    name=ap.parse_args().sample
    case,step,layer=SPEC[name]
    target=DATA/f'{name}.pt'
    assert not target.exists(), target
    source=Path('/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases')/case/'bf16_teacher_calls.pt'
    calls=torch.load(source,map_location='cpu',weights_only=False)
    assert len(calls)==20
    call=calls[step]
    del calls
    pipe=load_h3_pipeline(full=False,vram_limit_gib=18.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    ids=call['kwargs']['img_pos_info']['position_ids'].reshape(-1).long()
    assert ids.numel()==37*18*32
    prefix=int(ids[0])
    assert torch.equal(ids,torch.arange(prefix,prefix+ids.numel()))
    active=[False]
    hook=pipe.dit.blocks[layer].attn.register_forward_pre_hook(lambda *args:active.__setitem__(0,True))
    original=comfy._sdpa_varlen_attention
    packet={}
    def intercept(q,k,v,cu_seqlens,softmax_scale):
        cu,scale=cu_seqlens,softmax_scale
        if not active[0]:
            return original(q,k,v,cu,scale)
        bounds=cu.detach().cpu().tolist()
        n=bounds[1]
        assert n==prefix+ids.numel() and q.shape==k.shape==v.shape
        assert q.shape[1:]==(56,128) and q.dtype==torch.bfloat16
        packet.update(q=q[:n].transpose(0,1).unsqueeze(0).cpu().contiguous(),
                      k=k[:n].transpose(0,1).unsqueeze(0).cpu().contiguous(),
                      v=v[:n].transpose(0,1).unsqueeze(0).cpu().contiguous(),
                      video_grid=[37,18,32],prefix_tokens=prefix,layer=layer,step=step,
                      case=case,source=str(source),softmax_scale=float(scale),bounds=bounds,
                      capture_scope='post-norm/RoPE main sequence; independent padding segment excluded; BF16 model and trajectory')
        raise Captured()
    comfy._sdpa_varlen_attention=intercept
    started=time.monotonic()
    try:
        try:
            pipe.dit(*tree_device(call['args'],'cuda'),**tree_device(call['kwargs'],'cuda'))
        except Captured:
            pass
        else:
            raise RuntimeError('requested attention not captured')
    finally:
        hook.remove()
        comfy._sdpa_varlen_attention=original
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
        torch.save(packet,f)
    print(json.dumps({'sample':name,'path':str(target),'shape':list(packet['q'].shape),
                      'prefix':prefix,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    main()
