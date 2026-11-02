#!/usr/bin/env python3
"""Measure official DraftMap top-k recall against same-QKV dense block oracles."""
import argparse, hashlib, importlib, json, math, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs
from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_h3_draft_probability
HEADS=(0,8,16,24,32,40,48,55);LAYERS=(0,12,24,36,49);RATES=(.1,.2,.3,.5);BLOCK=128
OUT=ROOT/'results/research/draftmap_hit_20261102';OUT.mkdir(parents=True,exist_ok=True)
@torch.inference_mode()
def measure(q,k,v,prefix,grid):
 # Inputs [B,H,T,D] with heads already selected, BF16 and original contiguous prefix.
 q,k,v=(x.contiguous() for x in (q,k,v));device=q.device
 layout=materialize_ragged_2d_layout(device,frames=grid[0],height=grid[1],width=grid[2],logical_block=BLOCK,enable_anchors=False)
 nblocks=layout.counts.numel();counts=layout.counts.view(1,-1).contiguous()
 prepared=_prepare_h3_sm120_inputs(q,k,v,video_token_indices=layout.indices,video_slot_valid=layout.slot_valid,video_valid_counts=layout.counts,prefix_tokens=prefix,query_block_size=BLOCK,phases=('int8',),has_prefix_query_int8=True,has_pool=True,has_maxpool=True,global_scales=None)
 qp,kp=prepared[:2];qmax,kmax=prepared[-2:]
 proxy=sm120_h3_draft_probability(qp,kp,qmax,kmax,maxpool_weight=.2).float()[0]
 del prepared,qp,kp,qmax,kmax
 # Gather BF16 in the exact official ragged video order. Prefix remains in denominator.
 indices=layout.indices+prefix;valid=layout.slot_valid;h=q.shape[1];b=nblocks
 qv=q[0,:,indices,:].contiguous().view(h,b,BLOCK,128)
 kv=k[0,:,indices,:].contiguous();kv[:,~valid,:]=0
 keys=torch.cat((k[0,:,:prefix,:],kv),dim=1).contiguous()
 key_valid=torch.cat((torch.ones(prefix,device=device,dtype=torch.bool),valid))
 masses=torch.empty((h,b,b),device=device,dtype=torch.float32);maxima=torch.empty_like(masses)
 for qb in range(b):
  count=int(layout.counts[qb]);qrow=qv[:,qb,:count,:].contiguous()
  # BF16 GEMM (FP32 accumulation, BF16 materialized logits), FP32 softmax.
  logits=torch.bmm(qrow,keys.transpose(1,2)).float().mul_(128**-.5)
  logits.masked_fill_(~key_valid.view(1,1,-1),-torch.inf)
  vid=logits[:,:,prefix:].view(h,count,b,BLOCK)
  maxima[:,qb,:]=vid.amax(dim=(1,3))
  prob=logits.softmax(dim=-1)[:,:,prefix:].view(h,count,b,BLOCK)
  masses[:,qb,:]=prob.sum(dim=-1).mean(dim=1)
  del logits,vid,prob
 assert torch.isfinite(proxy).all() and torch.isfinite(masses).all() and torch.isfinite(maxima).all()
 scores={'proxy':proxy,'mass':masses,'qkmax':maxima};orders={name:torch.argsort(x.reshape(h,-1),dim=-1,descending=True,stable=True) for name,x in scores.items()};items=b*b
 result={'nblocks':b,'prefix_tokens':prefix,'video_tokens':math.prod(grid),'tile_count':items,'heads':list(HEADS),'budget':{}}
 for rate in RATES:
  n=min(items,max(1,math.floor(rate*items+.5))); chosen=orders['proxy'][:,:n];hits={}
  for key in ('mass','qkmax'):
   oracle=orders[key][:,:n];mask=torch.zeros((h,items),device=device,dtype=torch.bool);mask.scatter_(1,oracle,True);hits[key]=mask.gather(1,chosen).float().mean(dim=1).cpu().tolist()
  result['budget'][str(rate)]={'selected_per_head':n,'actual_fraction':n/items,'recall_by_head':hits}
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sample',help='existing real QKV for smoke');args=ap.parse_args()
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 if args.sample:
  p=torch.load(args.sample,map_location='cpu',weights_only=False,mmap=True);q,k,v=(p[x][:,HEADS].cuda() for x in ('q','k','v'));r=measure(q,k,v,p['prefix_tokens'],tuple(p['video_grid']));(OUT/'smoke.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return
 from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
 from diffsynth.models import minimax_h3_dit_comfy as comfy
 importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
 row=read_prompt(16);pipe=load_h3_pipeline(full=True,vram_limit_gib=18.);pipe.load_models_to_device(['dit']);pipe.dit.eval();state={'step':-1,'layer':None};handles=[];out=OUT/'per_cell.jsonl'
 if out.exists():raise FileExistsError(out)
 def bind(module,args,kwargs):
  state['step']+=1;ids=kwargs['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long();state['prefix']=int(ids[0]);assert ids.numel()==37*15*26 and torch.equal(ids,torch.arange(state['prefix'],state['prefix']+ids.numel()))
 handles.append(pipe.dit.register_forward_pre_hook(bind,with_kwargs=True))
 for layer in LAYERS:
  def enter(_m,_a,layer=layer):state['layer']=layer
  def leave(_m,_a,_o):state['layer']=None
  handles.extend([pipe.dit.blocks[layer].attn.register_forward_pre_hook(enter),pipe.dit.blocks[layer].attn.register_forward_hook(leave)])
 original=comfy._sdpa_varlen_attention
 def intercept(q,k,v,cu_seqlens,softmax_scale):
  layer=state['layer'];step=state['step']
  if layer in LAYERS:
   bounds=cu_seqlens.detach().cpu().tolist();n=bounds[1];assert n==state['prefix']+37*15*26 and softmax_scale==128**-.5
   qs,ks,vs=(x[:n,HEADS,:].transpose(0,1).unsqueeze(0).contiguous() for x in (q,k,v));start=time.monotonic();r=measure(qs,ks,vs,state['prefix'],(37,15,26));r.update(step=step,layer=layer,seconds=time.monotonic()-start)
   with out.open('a') as f:f.write(json.dumps(r)+'\n')
   print('MEASURED',step,layer,round(r['seconds'],2),flush=True)
  return original(q,k,v,cu_seqlens,softmax_scale)
 comfy._sdpa_varlen_attention=intercept
 class Done(Exception):pass
 def forward_hook(_m,_a,_o):
  if state['step']==19:raise Done()
 handles.append(pipe.dit.register_forward_hook(forward_hook))
 try:
  try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
  except Done:pass
  else:raise RuntimeError('last step was not reached')
 finally:
  comfy._sdpa_varlen_attention=original
  for h in handles:h.remove()
 rows=[json.loads(x) for x in out.read_text().splitlines()];assert {(r['step'],r['layer']) for r in rows}=={(t,l) for t in range(20) for l in LAYERS}
 (OUT/'capture_manifest.json').write_text(json.dumps({'status':'complete','prompt_id':16,'seed':int(row['seed']),'height':480,'width':832,'frames':124,'steps':20,'layers':LAYERS,'heads':HEADS,'rows':len(rows),'block':128,'maxpool_weight':.2,'trajectory':'BF16 dense closed-loop','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print('ALL_DONE',flush=True)
if __name__=='__main__':main()
