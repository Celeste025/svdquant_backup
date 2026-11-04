#!/usr/bin/env python3
import json,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'third_party/anemoi-code')]
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs,_route_sm120_probability
from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_h3_draft_probability
@torch.inference_mode()
def main():
 p=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
 q,k,v=[p[x][:,:4].cuda().contiguous() for x in ('q','k','v')];g=p['video_grid'];lay=materialize_ragged_2d_layout(q.device,frames=g[0],height=g[1],width=g[2],logical_block=64,enable_anchors=False)
 prep=_prepare_h3_sm120_inputs(q,k,v,video_token_indices=lay.indices,video_slot_valid=lay.slot_valid,video_valid_counts=lay.counts,prefix_tokens=p['prefix_tokens'],query_block_size=64,phases=('int8',),has_prefix_query_int8=True,has_pool=True,has_maxpool=True,global_scales=None)
 proxy=sm120_h3_draft_probability(prep[0],prep[1],prep[-2],prep[-1],maxpool_weight=.2);outs=[]
 for skip,nv,i8 in ((.8,.5,.5),(.5,.8,.2),(0.,.9,.1)):
  plan=_route_sm120_probability(proxy,None,None,anchor_count=0,sparsity_ratio=skip,ratios=(nv,i8,0.,0.));ids=plan.block_ids.long();cols=proxy.shape[-1];pos=torch.arange(cols,device=q.device).view(1,1,1,cols);n4=plan.nvfp4_counts.unsqueeze(-1);n8=plan.fp8_counts.unsqueeze(-1);val=torch.where(pos<n4,1,torch.where(pos<n4+n8,2,0)).to(torch.uint8);dst=torch.full((*proxy.shape[:-1],cols+1),0,dtype=torch.uint8,device=q.device);dst.scatter_(-1,torch.where(val==0,cols,ids),val);outs.append(dst[...,:cols]);print(skip,int((dst==2).sum()),flush=True)
 assert torch.equal(outs[0]==2,outs[1]==2) and torch.equal(outs[1]==2,outs[2]==2)
 assert ((outs[0]>0)&(outs[1]==0)).sum()==0 and ((outs[1]>0)&(outs[2]==0)).sum()==0
 (ROOT/'results/research/E214/route_nesting_check.json').write_text(json.dumps({'status':'complete','capture':'E187 race step 0 layer 25','heads_checked':4,'int8_selection_identical':True,'retained_support_nested':True,'block':64},indent=2)+'\n');print('PASS',flush=True)
if __name__=='__main__':main()
