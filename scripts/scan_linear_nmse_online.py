#!/usr/bin/env python3
"""Online same-input Linear error scan over BF16 denoising trajectories.

No activations are archived.  At each selected BF16 transformer call, every
Linear forward hook immediately evaluates its matching PTQ module and keeps only
scalar error accumulators keyed by prompt, seed, timestep, branch and layer.
"""
from __future__ import annotations

import argparse, csv, gc, importlib.util, json, math, os
from collections import defaultdict
from pathlib import Path
import torch
from linear_quant_context import (
 UnsupportedExternalTransform, audit_isolated_linear_context, prepare_isolated_linear_input,
)

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/data/models/svdquant-wjq')
PROMPTS=[
 'A red fox walking through a snowy forest at sunrise, cinematic natural light.',
 'A yellow school bus driving along a winding mountain road, wide cinematic view.',
 'A close-up of ocean waves crashing against black volcanic rocks at dusk.',
 'A small robot watering flowers in a bright glass greenhouse, detailed illustration.',
]

def module(name):
 p=ROOT/'scripts'/name; s=importlib.util.spec_from_file_location(p.stem,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def tensor(x):
 return x if torch.is_tensor(x) else x[0] if isinstance(x,(tuple,list)) else x.sample
def targets(n,k):
 if k <= 1: return [0]
 return list(range(n)) if n<=k else sorted({round(i*(n-1)/(k-1)) for i in range(k)})
def kind(n):
 n=n.lower()
 if any(x in n for x in ('to_q','to_k','to_v','q_proj','k_proj','v_proj')): return 'attn_qkv'
 if 'attn' in n and any(x in n for x in ('to_out','out_proj','proj_out')): return 'attn_out'
 if any(x in n for x in ('ffn','.ff.','.mlp.','net.')): return 'ffn'
 if any(x in n for x in ('embedder','time_embed','context_embed')): return 'condition_time'
 return 'other'

@torch.inference_mode()
def run(model, out, num_prompts, num_tsteps):
 os.environ['DATA_ROOT']=str(DATA)
 if model=='wan2.1-1.3b':
  h=module('infer_wan_bf16_vs_w4a4_one.py'); bf=h.load_bf16_pipeline(); ck=DATA/'ckpts/wan2.1-1.3b-real-nvfp4-s16'; qp=h.load_quant_pipeline(ck); steps=h.STEPS
  def kwargs(prompt,seed): return dict(prompt=prompt,negative_prompt=h.NEGATIVE,height=h.HEIGHT,width=h.WIDTH,num_frames=h.NUM_FRAMES,num_inference_steps=steps,guidance_scale=h.GUIDANCE,generator=torch.Generator('cuda').manual_seed(seed),output_type='latent')
 elif model=='rcm-wan2.1-1.3b':
  from diffusers import WanPipeline
  h=module('infer_rcm_wan_4step.py'); base=DATA/'models/rcm-Wan2.1-T2V-1.3B-Diffusers'; ck=DATA/'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16'; bf=WanPipeline.from_pretrained(base,torch_dtype=torch.bfloat16).to('cuda'); bf.to('cpu'); gc.collect(); torch.cuda.empty_cache(); qp=WanPipeline.from_pretrained(base,torch_dtype=torch.bfloat16).to('cuda'); h.load_quantized_transformer(qp,ck,base); qp.transformer.to('cpu'); bf.to('cuda'); steps=4
  def kwargs(prompt,seed): return None
 elif model.startswith('flux'):
  name,steps,guidance=('FLUX.1-dev',50,3.5) if model.endswith('dev') else ('FLUX.1-schnell',4,0.)
  os.environ.update({'FLUX_MODEL_PATH':str(DATA/'models'/name),'FLUX_MODEL_CONFIG':f'configs/model/{name.lower()}.yaml','FLUX_NUM_STEPS':str(steps),'FLUX_GUIDANCE':str(guidance),'FLUX_CALIB_PATH':str(DATA/'datasets/torch.bfloat16'/name.lower()/('fmeuler4-g0' if not guidance else 'fmeuler50-g3.5')/'qdiff'/'s64')})
  h=module('infer_bf16_vs_w4a4_one.py'); bf=h.load_bf16_pipeline(); bf.to('cpu'); gc.collect(); torch.cuda.empty_cache(); ck=DATA/'ckpts'/f'{name.lower()}-int4-fast-s64-lowmem'; qp=h.load_quant_pipeline(ck); qp.transformer.to('cpu'); bf.to('cuda')
  def kwargs(prompt,seed): return dict(prompt=prompt,height=1024,width=1024,num_inference_steps=steps,guidance_scale=guidance,generator=torch.Generator('cuda').manual_seed(seed),output_type='latent')
 else: raise ValueError('rCM-Wan will use its explicit 4-step driver; run Wan/Flux first')
 qp.transformer.to('cpu'); gc.collect(); torch.cuda.empty_cache()
 qmods=dict(qp.transformer.named_modules()); acc=defaultdict(lambda:[0.,0.,0,0.,0.,0.]); chosen=targets(steps,num_tsteps)
 audit={}
 for name, fm in bf.transformer.named_modules():
  qm=qmods.get(name)
  if name and isinstance(fm,torch.nn.Linear) and isinstance(qm,torch.nn.Linear):
   audit[name]=audit_isolated_linear_context(qp.transformer,name,qm.in_features)
 supported=sum(v['status']=='supported' for v in audit.values())
 print(f'[{model}] isolated-input audit: {supported}/{len(audit)} matching Linears supported',flush=True)
 state={'step':0,'branch':0,'active':None}; handles=[]
 def root_pre(_m,_a,kw): state['active']=(state['step'],state['branch']) if state['step'] in chosen else None
 def root_post(_m,_a,_o): state['branch']+=1
 handles += [bf.transformer.register_forward_pre_hook(root_pre,with_kwargs=True),bf.transformer.register_forward_hook(root_post)]
 for name, fm in bf.transformer.named_modules():
  if not name or not isinstance(fm,torch.nn.Linear): continue
  qm=qmods.get(name)
  if qm is None: continue
  def hook(_m,args,y,layer=name,q=qm):
   active=state['active']
   if active is None or not args or not torch.is_tensor(args[0]): return
   x=args[0]; ref=tensor(y)
   try:
    q.to('cuda'); qi=x; w=getattr(q,'weight',None)
    qi,context=prepare_isolated_linear_input(qp.transformer,layer,qi)
    if torch.is_tensor(w) and w.is_floating_point(): qi=qi.to(dtype=w.dtype)
    pred=tensor(q(qi)); e=(pred.float()-ref.float()); key=(state['prompt'],state['seed'],active[0],active[1],layer)
    a=acc[key]; a[0]+=float(e.square().sum()); a[1]+=float(ref.float().square().sum()); a[2]+=ref.numel(); a[3]+=float((pred.float()*ref.float()).sum()); a[4]+=float(pred.float().square().sum()); a[5]+=float(ref.float().square().sum()); audit[layer]={'status':'supported','input_context':context.mode}
   except UnsupportedExternalTransform as exc:
    audit[layer]={'status':'unsupported','reason':str(exc)}
   except Exception as exc:
    audit[layer]={'status':'failed','reason':f'{type(exc).__name__}: {exc}'}
  handles.append(fm.register_forward_hook(hook))
 for pi,prompt in enumerate(PROMPTS[:num_prompts]):
  state.update(prompt=pi,seed=42+pi,step=0,branch=0,active=None)
  def callback(_pipe,i,_t,kw): state.update(step=i+1,branch=0,active=None); return kw
  print(f'[{model}] BF16 trajectory prompt {pi+1}/{num_prompts}',flush=True)
  if model=='rcm-wan2.1-1.3b':
   dev=torch.device('cuda'); dt=bf.transformer.dtype; emb,_=bf.encode_prompt(prompt=prompt,do_classifier_free_guidance=False,max_sequence_length=512,device=dev,dtype=dt); lat=bf.prepare_latents(1,bf.transformer.config.in_channels,480,832,77,torch.float32,dev,torch.Generator(dev).manual_seed(42+pi)).to(torch.float64); ts=torch.tensor([math.atan(80),1.5,1.4,1.0,0.0],device=dev,dtype=torch.float64); ts=torch.sin(ts)/(torch.cos(ts)+torch.sin(ts)); lat=lat*ts[0]
   for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    state.update(step=i,branch=0,active=None); vel=bf.transformer(hidden_states=lat.to(dt),timestep=torch.tensor([tc*1000],device=dev,dtype=dt),encoder_hidden_states=emb,return_dict=False)[0].to(torch.float64); lat=(1-tn)*(lat-tc*vel)+tn*torch.randn(lat.shape,dtype=torch.float32,device=dev,generator=torch.Generator(dev).manual_seed(4200+pi*10+i))
  else: bf(**kwargs(prompt,42+pi),callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents'])
 for hnd in handles: hnd.remove()
 qp.transformer.to('cpu'); gc.collect(); torch.cuda.empty_cache()
 rows=[]
 for (pi,seed,step,branch,name),(se,sr,n,dot,sp,sr2) in acc.items(): rows.append({'model':model,'prompt_id':pi,'seed':seed,'timestep_index':step,'cfg_branch':branch,'layer':name,'layer_type':kind(name),'input_context':audit.get(name,{}).get('input_context','unknown'),'sum_sq_err':se,'sum_ref_sq':sr,'numel':n,'mse':se/n,'nmse':se/max(sr,1e-30),'cosine':dot/max((sp*sr2)**.5,1e-30)})
 out.mkdir(parents=True,exist_ok=True); jp=out/f'{model}_online_records.json'; cp=out/f'{model}_online_records.csv'; jp.write_text(json.dumps({'model':model,'n_prompts':num_prompts,'selected_timestep_indices':chosen,'definition':'All-token BF16 output versus the matching PTQ Linear evaluated in its reconstructed quantized input coordinate. Direct Linear hooks (activation QDQ and low-rank compensation) remain active. Unsupported ancestor transforms are excluded rather than silently bypassed.','context_audit':audit,'records':rows},indent=2)+'\n')
 with cp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 print(f'[{model}] wrote {len(rows)} online records -> {jp}',flush=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=('wan2.1-1.3b','flux.1-dev','flux.1-schnell','rcm-wan2.1-1.3b')); ap.add_argument('--num-prompts',type=int,default=4); ap.add_argument('--num-timesteps',type=int,default=10); ap.add_argument('--out-dir',type=Path,default=ROOT/'results/reports/linear_online_4x_timestep'); a=ap.parse_args(); run(a.model,a.out_dir.resolve(),a.num_prompts,a.num_timesteps)
if __name__=='__main__': main()
