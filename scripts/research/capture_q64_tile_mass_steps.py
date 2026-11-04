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
OUT=ROOT/'results/research/tile_probability_coverage_q64_20261103'
TOP=ROOT/'results/research/tile_probability_overlap_q64_20261103'
OUT.mkdir(parents=True,exist_ok=True)
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
   lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=64,enable_anchors=False)
   qh,kh,vh=(x[:n].transpose(0,1).unsqueeze(0).contiguous() for x in (q,k,v))
   nb=lay.counts.numel();counts=lay.counts;valid=lay.slot_valid
   keyvalid=torch.cat((torch.ones(state['prefix'],device=q.device,dtype=torch.bool),valid))
   mass=torch.empty((56,nb,nb),device=q.device,dtype=torch.float32)
   prefix_mass=torch.empty((56,nb),device=q.device,dtype=torch.float32)
   for begin in range(0,56,4):
    end=min(begin+4,56);h=end-begin
    qv=qh[0,begin:end,lay.indices+state['prefix']].float().reshape(h,nb,64,128)
    keys=torch.cat((kh[0,begin:end,:state['prefix']].float(),kh[0,begin:end,lay.indices+state['prefix']].float()),dim=1)
    for i in range(nb):
     nq=int(counts[i]);logits=torch.bmm(qv[:,i,:nq],keys.transpose(1,2))*128**-.5
     logits.masked_fill_(~keyvalid.view(1,1,-1),-torch.inf)
     prob=logits.softmax(-1);prefix_mass[begin:end,i]=prob[:,:,:state['prefix']].sum((1,2))
     vid=prob[:,:,state['prefix']:].view(h,nq,nb,64)
     mass[begin:end,i]=vid.sum((1,3))
   assert torch.isfinite(mass).all() and torch.isfinite(prefix_mass).all()
   row_total=mass.sum(-1)+prefix_mass
   assert torch.allclose(row_total,counts.float().view(1,-1).expand_as(row_total),rtol=2e-5,atol=2e-5)
   mean=mass/(counts.view(1,-1,1)*counts.view(1,1,-1))
   ref=np.load(TOP/f'top_step_{state["step"]:02d}.npz')['true']
   top=torch.argsort(mean.flatten(1),dim=-1,descending=True,stable=True)[:,:ref.shape[1]].to(torch.int32).cpu().numpy()
   assert np.array_equal(top,ref),f'true map changed at step {state["step"]}'
   np.savez_compressed(OUT/f'mass_step_{state["step"]:02d}.npz',mass=mass.cpu().numpy(),prefix_mass=prefix_mass.cpu().numpy(),counts=counts.cpu().numpy())
   maps.append(state['step'])
   print('MASS',state['step'],'blocks',nb,'selection_verified',flush=True)
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
 (OUT/'capture_manifest.json').write_text(json.dumps({'status':'complete','layer':25,'steps':20,'block':64,'prompt_id':16,'seed':int(row['seed']),'grid':grid,'prefix':state['prefix'],'heads':56,'trajectory':'BF16 dense','mass':'sum over valid query-key pairs, unnormalized across queries; prefix mass saved separately','top_map_replay_verified':True},indent=2)+'\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
