#!/usr/bin/env python3
"""Top-key interventions on same-input BF16 dense teacher attention."""
import argparse,importlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'third_party/anemoi-code')]
from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
OUT=ROOT/'results/research/E218';HEADS=(10,20,30);GRID=(37,15,26)
class Done(Exception):pass
@torch.inference_mode()
def measure(q,k,v,prefix,step,pid):
 lay=materialize_ragged_2d_layout(q.device,frames=GRID[0],height=GRID[1],width=GRID[2],logical_block=64,enable_anchors=False);nb=lay.counts.numel();idx=lay.indices.view(nb,64);valid=lay.slot_valid.view(nb,64);counts=lay.counts
 results=[]
 for h in HEADS:
  qh=q[:,h,:].float();kh=k[:,h,:].float();vh=v[:,h,:].float();length=qh.shape[0]-prefix;tile=torch.empty((nb,nb),device=q.device);tokenmass=torch.zeros(length,device=q.device)
  for i in range(nb):
   qi=qh[prefix+idx[i,:counts[i]]];prob=(qi@kh.T/128**.5).softmax(-1);vid=prob[:,prefix:];tokenmass+=vid.sum(0);gp=vid[:,idx.reshape(-1)].view(-1,nb,64);tile[i]=(gp*valid[None,:,:]).sum((0,2))/(int(counts[i])*counts)
  n=int(.1*nb*nb+.5);rank=torch.argsort(tile.flatten(),descending=True,stable=True)[:n];mask=torch.zeros(nb*nb,device=q.device,dtype=torch.bool);mask[rank]=True;degree=mask.view(nb,nb).float().mean(0);col=int(degree.argmax());positions=idx[col,:counts[col]];weighted=tokenmass[positions];order=torch.argsort(weighted,descending=True,stable=True);topids=positions[order[:5]]
  gen=torch.Generator(device='cpu').manual_seed(218000+pid*100+step*10+h);random_sets=[positions[torch.randperm(len(positions),generator=gen)[:5].to(q.device)] for _ in range(5)]
  # Top-5 in a deterministic randomly selected different block is an additional control.
  other=int(torch.randint(nb-1,(1,),generator=gen));other+=int(other>=col);otherids=idx[other,:counts[other]];other_top=otherids[torch.argsort(tokenmass[otherids],descending=True,stable=True)[:5]]
  sets=[topids,*random_sets,other_top,positions];labels=['top5','random_same0','random_same1','random_same2','random_same3','random_same4','other_block_top5','entire_block'];num=len(sets)
  numerr=torch.zeros(num,device=q.device,dtype=torch.float64);den=torch.zeros((),device=q.device,dtype=torch.float64);removed=torch.zeros(num,device=q.device,dtype=torch.float64);max_err=0.
  for start in range(0,length,384):
   qq=qh[prefix+start:prefix+min(start+384,length)];logits=qq@kh.T/128**.5;prob=logits.softmax(-1);ref=prob@vh;den+=ref.double().square().sum()
   for j,ids in enumerate(sets):
    sel=ids+prefix;weights=prob[:,sel];mass=weights.sum(-1);contrib=weights@vh[sel];modified=(ref-contrib)/(1-mass).clamp_min(1e-12)[:,None]
    numerr[j]+=(modified-ref).double().square().sum();removed[j]+=mass.double().sum()
    if start==0 and j==0:
     masked=logits.clone();masked[:,sel]=-torch.inf;direct=masked.softmax(-1)@vh;max_err=float((modified-direct).abs().max());assert torch.allclose(modified,direct,rtol=2e-4,atol=2e-5),(pid,step,h,max_err)
  deg=degree.cpu().numpy();frames=np.arange(nb)//lay.blocks_per_frame;npdeg=deg.reshape(37,7)
  interventions={name:{'nmse':float(numerr[j]/den),'mean_removed_mass':float(removed[j]/length)} for j,name in enumerate(labels)}
  topcpu=topids.cpu().tolist();results.append({'head':h,'strongest_block':col,'strongest_block_latent_time':col//7,'strongest_block_spatial_region':col%7,'selection_query_fraction':float(degree[col]),'degree_for_first_block':float(degree[0]),'first_block_rank':int((degree>degree[0]).sum())+1,'degree_by_temporal_mod5':[float(deg[frames%5==i].mean()) for i in range(5)],'block_degree_by_time_region':npdeg.tolist(),'top5_latent_flat_indices':topcpu,'top5_spatial_coords':[[int(i//390),int((i%390)//26),int(i%26)] for i in topcpu],'top5_fraction_of_block_mass':float(weighted[order[:5]].sum()/weighted.sum()),'random_block':other,'interventions':interventions,'direct_mask_validation_max_abs':max_err})
  print('HEAD',pid,step,h,'block',col,'top5nmse',interventions['top5']['nmse'],flush=True)
 return {'status':'complete','prompt_id':pid,'step':step,'layer':25,'grid':list(GRID),'prefix_tokens':prefix,'heads':results,'notes':'top5 selected using aggregated attention probabilities on same QKV; each intervention removes selected video keys and recomputes exact softmax renormalization; all video queries; per-head output NMSE against FP32 direct-softmax reference. Controls 5 seeded random 5-token subsets in same block; additional top5 of one random distinct key block; FP32 offline on fixed QKV, no denoising trajectory change.'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prompt',type=int,required=True);pid=ap.parse_args().prompt;torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch';row=read_prompt(pid);pipe=load_h3_pipeline(full=True,vram_limit_gib=18.);pipe.load_models_to_device(['dit']);pipe.dit.eval();state={'step':-1,'inlayer':False};hooks=[]
 def bind(_m,_a,kw):
  state['step']+=1;ids=kw['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long();state['prefix']=int(ids[0]);assert ids.numel()==37*15*26 and torch.equal(ids,torch.arange(state['prefix'],state['prefix']+ids.numel()))
 def enter(*_):state['inlayer']=True
 def leave(*_):state['inlayer']=False
 hooks.extend([pipe.dit.register_forward_pre_hook(bind,with_kwargs=True),pipe.dit.blocks[25].attn.register_forward_pre_hook(enter),pipe.dit.blocks[25].attn.register_forward_hook(leave)]);original=comfy._sdpa_varlen_attention
 def intercept(q,k,v,cu_seqlens,softmax_scale):
  if state['inlayer'] and state['step'] in (0,10):
   n=cu_seqlens.detach().cpu().tolist()[1];assert n==state['prefix']+37*15*26 and softmax_scale==128**-.5
   d=measure(q[:n],k[:n],v[:n],state['prefix'],state['step'],pid);d['seed']=int(row['seed']);(OUT/f'p{pid:03d}_step{state["step"]}.json').write_text(json.dumps(d,indent=2)+'\n');print('STEP_COMPLETE',pid,state['step'],flush=True)
   if state['step']==10:raise Done()
  return original(q,k,v,cu_seqlens,softmax_scale)
 comfy._sdpa_varlen_attention=intercept
 try:
  try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
  except Done:pass
  else:raise RuntimeError('target step not reached')
 finally:
  comfy._sdpa_varlen_attention=original
  for h in hooks:h.remove()
 print('COMPLETE',pid,flush=True)
if __name__=='__main__':main()
