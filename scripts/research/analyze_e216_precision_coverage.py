#!/usr/bin/env python3
"""Measure route coverage of Q/K blocks and dense mixed QKV preparation cost."""
import json,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/anemoi-code'))
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs,_route_sm120_probability,_nvfp4_global_scales
from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_h3_draft_probability
OUT=ROOT/'results/research/E216';MANIFEST=ROOT/'results/research/E187/manifest.json'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 entries=json.loads(MANIFEST.read_text())['captures'];result={'status':'running','scope':'original DraftMap Q64, no skipped video tiles, per-head global precision allocation; input BF16, prefix not included in row/col coverage; step/layer zero-based','samples':[]}
 for e in entries:
  d=torch.load(e['path'],map_location='cpu',weights_only=True,mmap=True);q,k,v=[d[z].cuda().contiguous() for z in ('q','k','v')];grid=d['video_grid'];prefix=d['prefix_tokens'];lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=64,enable_anchors=False);nb=lay.counts.numel()
  kw={'video_token_indices':lay.indices,'video_slot_valid':lay.slot_valid,'video_valid_counts':lay.counts,'prefix_tokens':prefix,'query_block_size':64,'has_prefix_query_int8':True,'has_pool':True,'has_maxpool':True}
  prep=_prepare_h3_sm120_inputs(q,k,v,phases=('int8',),global_scales=None,**kw);proxy=sm120_h3_draft_probability(prep[0],prep[1],prep[-2],prep[-1],maxpool_weight=.2);del prep
  sample={'step':e['step'],'layer':e['layer'],'capture_sha256':e['sha256'],'blocks_per_axis':nb,'fractions':{}};output={}
  for frac in (.1,.2,.3):
   plan=_route_sm120_probability(proxy,None,None,anchor_count=0,sparsity_ratio=0.,ratios=(1-frac,frac,0.,0.));ids=plan.block_ids.long();cols=nb;pos=torch.arange(cols,device=q.device).view(1,1,1,cols);n4=plan.nvfp4_counts.unsqueeze(-1);n8=plan.fp8_counts.unsqueeze(-1);val=torch.where(pos<n4,1,torch.where(pos<n4+n8,2,0)).to(torch.uint8);dst=torch.zeros((*proxy.shape[:-1],cols+1),dtype=torch.uint8,device=q.device);dst.scatter_(-1,torch.where(val==0,cols,ids),val);route=dst[0,...,:cols];assert ((route==0).sum()==0)
   data={}
   for axis,name in ((-1,'query'),(-2,'key_value')):
    count=(route==2).sum(axis);n4count=(route==1).sum(axis);assert ((count+n4count)==nb).all()
    data[name]={'both':int(((count>0)&(n4count>0)).sum()),'int8_only':int((count==nb).sum()),'nvfp4_only':int((count==0).sum()),'total':count.numel(),'int8_incident_fraction_by_head':(count>0).float().mean(-1).cpu().tolist(),'both_fraction_by_head':((count>0)&(n4count>0)).float().mean(-1).cpu().tolist()}
   sample['fractions'][str(frac)]=data;output[f'route_{int(frac*100)}']=route.cpu()
  torch.save(output,OUT/f'route_step{e["step"]}_layer{e["layer"]}.pt');result['samples'].append(sample);print('ROUTES',e['step'],e['layer'],flush=True)
  if (e['step'],e['layer'])==(0,25):
   cal=json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales'];gs=_nvfp4_global_scales(q.device,tuple(cal[n][25] for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
   arms={'both':(('nvfp4','int8'),gs),'int8_video_only':(('int8',),None),'nvfp4_only_no_prefix_int8':(('nvfp4',),gs)};samples={k:[] for k in arms};bytes0={}
   for turn in range(2):
    for name in (list(arms) if turn==0 else list(arms)[::-1]):
     phases,scales=arms[name];kw['has_prefix_query_int8']=(name!='nvfp4_only_no_prefix_int8')
     for i in range(13):
      start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();out=_prepare_h3_sm120_inputs(q,k,v,phases=phases,global_scales=scales,**kw);end.record();end.synchronize()
      if name not in bytes0:
       def size(x):
        if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
        if isinstance(x,(tuple,list)):return sum(size(t) for t in x)
        return 0
       bytes0[name]={'nvfp4_operands':size(out[5]),'int8_operands':size(out[7]),'prefix_int8':size(out[8]),'all_outputs':size(out)}
      if i>=3:samples[name].append(start.elapsed_time(end))
      del out
   result['preparation']={'scope':'isolated same-input fused QKV preparation, includes mean/max pooling; prefix INT8 prepared only in INT8-active arms (kernel requires INT8 operands); CUDA event, 2 orders, each 3 warmup + 10 samples; toggling fused branches changes kernel configuration; differences not guaranteed end-to-end savings','gpu':torch.cuda.get_device_name(),'sample_step':0,'sample_layer':25,'samples_ms':samples,'mean_ms':{k:statistics.mean(v) for k,v in samples.items()},'outputs_bytes':bytes0};print('PREP',result['preparation']['mean_ms'],flush=True)
 result['status']='complete';(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':main()
