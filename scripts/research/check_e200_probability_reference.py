#!/usr/bin/env python3
"""Check captured INT8 softmax tile means against direct real-QKV calculation."""
import math,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/anemoi-code'))
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs, _expand_q128_valid_k
from anemoi.layers.attention.mpa.probability_capture import int8_softmax_mean
p=Path('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt')
d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);q,k,v=(d[x][:,:2].cuda().contiguous() for x in ('q','k','v'));grid=tuple(d['video_grid']);prefix=int(d['prefix_tokens']);lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=128,enable_anchors=False)
pb=math.ceil(prefix/64);pc=(prefix-torch.arange(pb,device=q.device,dtype=torch.int32)*64).clamp(0,64).view(1,-1);vc=lay.counts.view(1,-1);valid=_expand_q128_valid_k(pc,vc)
prep=_prepare_h3_sm120_inputs(q,k,v,video_token_indices=lay.indices,video_slot_valid=lay.slot_valid,video_valid_counts=lay.counts,prefix_tokens=prefix,query_block_size=128,phases=('int8',),has_prefix_query_int8=True,has_pool=False,has_maxpool=False,global_scales=None)
operands=prep[7];q8,qs,k8,ks,*_=operands;got=int8_softmax_mean(operands,valid,lay.counts,pb).float()/1024
keys=k8.float()*ks.repeat_interleave(64,-1).unsqueeze(-1);mask=(torch.arange(64,device=q.device)[None,:]<valid[0,:,None]).flatten()
max_error=0.
for i in (0,lay.counts.numel()//2,lay.counts.numel()-1):
 n=int(lay.counts[i]);queries=q8[:,:,i*128:i*128+n].float()*qs[:,:,i,None,None];logits=queries@keys.transpose(-1,-2)/math.sqrt(q8.shape[-1]);logits.masked_fill_(~mask.view(1,1,1,-1),-torch.inf);prob=logits.softmax(-1)
 for j in range(lay.counts.numel()):
  nk=int(lay.counts[j]);start=(pb+2*j)*64;ref=prob[:,:,:,start:start+nk].mean((-1,-2));actual=got[:,:,i,j];torch.testing.assert_close(actual,ref,rtol=.003,atol=2e-7);max_error=max(max_error,(actual-ref).abs().max().item())
print('REAL_INT8_REFERENCE_OK heads=2 query_tiles=3 keys=all max_abs_error=',max_error,flush=True)
