#!/usr/bin/env python3
"""Four-call teacher-forced rCM output error: NVFP4/INT4 versus BF16."""
from __future__ import annotations
import argparse, gc, json, math
from pathlib import Path
import torch
from diffusers import WanPipeline, WanTransformer3DModel
from infer_rcm_wan_4step import load_quantized_transformer

ROOT=Path(__file__).resolve().parents[1]; DATA=Path('/data1/models/svdquant-wjq')
BASE=DATA/'models/Wan2.1-T2V-1.3B-Diffusers'; RCM=DATA/'models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer'
PROMPT='A professional skateboarder performing a high-speed kickflip down a concrete stair set, dynamic tracking shot.'
def cpu(x): return x.detach().cpu()
def metric(x,y):
 x=x.float(); y=y.float(); d=x-y
 return {'max_abs':float(d.abs().max()),'mse':float(d.square().mean()),'rmse':float(d.square().mean().sqrt()),'nmse':float(d.square().sum()/y.square().sum().clamp_min(1e-30)),'cosine':float((x.flatten()@y.flatten())/(x.flatten().norm()*y.flatten().norm()).clamp_min(1e-30))}
@torch.inference_mode()
def bf_trace(seed):
 pipe=WanPipeline.from_pretrained(BASE,torch_dtype=torch.bfloat16); pipe.transformer=WanTransformer3DModel.from_pretrained(RCM,torch_dtype=torch.bfloat16); pipe=pipe.to('cuda'); emb,_=pipe.encode_prompt(prompt=PROMPT,do_classifier_free_guidance=False,max_sequence_length=512,device='cuda',dtype=pipe.transformer.dtype); pipe.text_encoder.to('cpu')
 gen=torch.Generator('cuda').manual_seed(seed); lat=pipe.prepare_latents(batch_size=1,num_channels_latents=pipe.transformer.config.in_channels,height=480,width=832,num_frames=77,dtype=torch.float32,device='cuda',generator=gen)
 trig=torch.tensor([math.atan(80.),1.5,1.4,1.,0.],dtype=torch.float64,device='cuda'); ts=torch.sin(trig)/(torch.cos(trig)+torch.sin(trig)); lat=lat.double()*ts[0]; trace=[]
 for t,n in zip(ts[:-1],ts[1:]):
  kw={'hidden_states':lat.to(torch.bfloat16),'timestep':torch.full((1,),t.float().item()*1000,device='cuda',dtype=torch.bfloat16),'encoder_hidden_states':emb,'return_dict':False}
  out=pipe.transformer(**kw)[0]; trace.append(( {k:cpu(v) if torch.is_tensor(v) else v for k,v in kw.items()},cpu(out) )); lat=(1-n)*(lat-t*out.double())+n*torch.randn(lat.shape,dtype=torch.float32,device='cuda',generator=gen)
 del pipe; gc.collect(); torch.cuda.empty_cache(); return trace
@torch.inference_mode()
def replay(trace,ckpt,recipe):
 pipe=WanPipeline.from_pretrained(BASE,torch_dtype=torch.bfloat16); pipe.transformer=WanTransformer3DModel.from_pretrained(RCM,torch_dtype=torch.bfloat16); pipe=pipe.to('cuda'); load_quantized_transformer(pipe,ckpt,BASE,'rcm-wan-int4-svdquant-v1' if recipe=='int4' else None); out=[]
 for kw,ref in trace: out.append(metric(pipe.transformer(**{k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in kw.items()})[0],ref.to('cuda')))
 del pipe; gc.collect(); torch.cuda.empty_cache(); return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); trace=bf_trace(a.seed); data={'seed':a.seed,'definition':'same BF16 trajectory input to BF16 and quantized rCM DiT output','nvfp4':replay(trace,DATA/'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16','nvfp4'),'int4':replay(trace,DATA/'ckpts/rcm-wan2.1-1.3b-int4-s16-g10','int4')}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__': main()
