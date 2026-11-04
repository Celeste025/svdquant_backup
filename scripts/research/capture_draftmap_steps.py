#!/usr/bin/env python3
"""Capture per-head native DraftMap tile scores along one dense H3 trajectory."""
import importlib, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs
from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_h3_draft_probability
OUT=ROOT/'results/research/tile_probability_overlap_20261103'
class Done(Exception): pass
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
 row=read_prompt(16);pipe=load_h3_pipeline(full=True,vram_limit_gib=18.);pipe.load_models_to_device(['dit']);pipe.dit.eval()
 state={'step':-1,'layer':None};handles=[];maps=[];grid=(37,15,26)
 def bind(_m,_a,kwargs):
  state['step']+=1;ids=kwargs['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long();state['prefix']=int(ids[0]);assert ids.numel()==37*15*26 and torch.equal(ids,torch.arange(state['prefix'],state['prefix']+ids.numel()))
 handles.append(pipe.dit.register_forward_pre_hook(bind,with_kwargs=True))
 def enter(*_):state['layer']=25
 def leave(*_):state['layer']=None
 handles.extend([pipe.dit.blocks[25].attn.register_forward_pre_hook(enter),pipe.dit.blocks[25].attn.register_forward_hook(leave)])
 original=comfy._sdpa_varlen_attention
 def intercept(q,k,v,cu_seqlens,softmax_scale):
  if state['layer']==25:
   n=cu_seqlens.detach().cpu().tolist()[1];assert n==state['prefix']+37*15*26 and softmax_scale==128**-.5 and len(maps)==state['step']
   lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=128,enable_anchors=False)
   qh,kh,vh=(x[:n].transpose(0,1).unsqueeze(0).contiguous() for x in (q,k,v))
   prep=_prepare_h3_sm120_inputs(qh,kh,vh,video_token_indices=lay.indices,video_slot_valid=lay.slot_valid,video_valid_counts=lay.counts,prefix_tokens=state['prefix'],query_block_size=128,phases=('int8',),has_prefix_query_int8=True,has_pool=True,has_maxpool=True,global_scales=None)
   m=sm120_h3_draft_probability(prep[0],prep[1],prep[-2],prep[-1],maxpool_weight=.2).float()[0].cpu().numpy()
   maps.append(m);np.savez_compressed(OUT/'draft_maps_partial.npz',maps=np.stack(maps))
   print('MAP',state['step'],m.shape,flush=True)
   if state['step']==19:raise Done()
  return original(q,k,v,cu_seqlens,softmax_scale)
 comfy._sdpa_varlen_attention=intercept
 try:
  try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
  except Done:pass
  else:raise RuntimeError('step 19 not reached')
 finally:
  comfy._sdpa_varlen_attention=original
  for h in handles:h.remove()
 assert len(maps)==20
 np.savez_compressed(OUT/'draft_maps.npz',maps=np.stack(maps))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
