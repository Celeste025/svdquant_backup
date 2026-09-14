#!/usr/bin/env python3
"""Capture first denoising transformer output via the established inference paths."""
from __future__ import annotations
import gc, importlib.util, json, math, os, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/data/models/svdquant-wjq')
OUT=ROOT/'results/reports/four_models_firststep_output_error.json'
PROMPT='A red fox walking through a snowy forest at sunrise, cinematic natural light.'
SEED=42
class StopFirst(Exception): pass

def load_script(name):
    p=ROOT/'scripts'/name
    s=importlib.util.spec_from_file_location(p.stem,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def value(x):
    if isinstance(x,tuple): x=x[0]
    elif hasattr(x,'sample'): x=x.sample
    return x.detach().float().cpu().clone()

@torch.no_grad()
def pipeline_first(pipe, kwargs):
    got=[]
    h=pipe.transformer.register_forward_hook(lambda _m,_a,o:got.append(value(o)))
    def cb(_p,i,_t,k):
        if i==0: raise StopFirst
        return k
    try:
        pipe(**kwargs, output_type='latent', callback_on_step_end=cb, callback_on_step_end_tensor_inputs=['latents'])
    except StopFirst: pass
    finally: h.remove()
    assert got
    return got

def clean(x):
    del x; gc.collect(); torch.cuda.empty_cache()
def metric(a,b):
    e=(b-a).square(); mse=e.mean(); return {'mse':float(mse),'nmse':float(mse/a.square().mean().clamp_min(1e-20)),'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),'shape':list(a.shape)}

def wan():
    m=load_script('infer_wan_bf16_vs_w4a4_one.py')
    kw=dict(prompt=PROMPT,negative_prompt=m.NEGATIVE,height=m.HEIGHT,width=m.WIDTH,num_frames=m.NUM_FRAMES,num_inference_steps=m.STEPS,guidance_scale=m.GUIDANCE,generator=torch.Generator('cuda').manual_seed(SEED))
    p=m.load_bf16_pipeline(); a=pipeline_first(p,kw); del p; gc.collect(); torch.cuda.empty_cache()
    p=m.load_quant_pipeline(DATA/'ckpts/wan2.1-1.3b-real-nvfp4-s16'); kw['generator']=torch.Generator('cuda').manual_seed(SEED); b=pipeline_first(p,kw); del p; gc.collect(); torch.cuda.empty_cache()
    assert len(a) == len(b) == 2, (len(a), len(b))
    u,c,qu,qc,g=a[0],a[1],b[0],b[1],m.GUIDANCE
    return {'settings':{'seed':SEED,'steps':m.STEPS,'guidance':g,'size':[m.HEIGHT,m.WIDTH],'frames':m.NUM_FRAMES},'outputs':{'uncond':metric(u,qu),'cond':metric(c,qc),'cfg':metric(u+g*(c-u),qu+g*(qc-qu))}}

def flux(name,steps,guidance):
    os.environ.update({'FLUX_MODEL_PATH':str(DATA/'models'/name),'FLUX_MODEL_CONFIG':f'configs/model/{name.lower()}.yaml','FLUX_NUM_STEPS':str(steps),'FLUX_GUIDANCE':str(guidance),'FLUX_CALIB_PATH':str(DATA/'datasets/torch.bfloat16'/name.lower()/(('fmeuler4-g0' if guidance==0 else 'fmeuler50-g3.5')+'/qdiff/s64'))})
    m=load_script('infer_bf16_vs_w4a4_one.py')
    kw=dict(prompt=PROMPT,height=1024,width=1024,num_inference_steps=steps,guidance_scale=guidance,generator=torch.Generator('cuda').manual_seed(SEED))
    p=m.load_bf16_pipeline(); a=pipeline_first(p,kw)[0]; del p; gc.collect(); torch.cuda.empty_cache()
    p=m.load_quant_pipeline(DATA/'ckpts'/(name.lower()+'-int4-fast-s64-lowmem')); kw['generator']=torch.Generator('cuda').manual_seed(SEED); b=pipeline_first(p,kw)[0]; del p; gc.collect(); torch.cuda.empty_cache()
    return {'settings':{'seed':SEED,'steps':steps,'guidance':guidance,'size':[1024,1024]},'outputs':{'noise_prediction':metric(a,b)}}

@torch.no_grad()
def rcm_one(pipe):
    dev=torch.device('cuda'); dt=pipe.transformer.dtype
    emb,_=pipe.encode_prompt(prompt=PROMPT,do_classifier_free_guidance=False,max_sequence_length=512,device=dev,dtype=dt)
    lat=pipe.prepare_latents(1,pipe.transformer.config.in_channels,480,832,77,torch.float32,dev,torch.Generator(dev).manual_seed(SEED)).to(torch.float64)
    t=math.sin(math.atan(80))/(math.cos(math.atan(80))+math.sin(math.atan(80)))
    return value(pipe.transformer(hidden_states=(lat*t).to(dt),timestep=torch.tensor([t*1000],device=dev,dtype=dt),encoder_hidden_states=emb,return_dict=False))

def rcm():
    from diffusers import WanPipeline
    model=DATA/'models/rcm-Wan2.1-T2V-1.3B-Diffusers'; helper=load_script('infer_rcm_wan_4step.py')
    p=WanPipeline.from_pretrained(model,torch_dtype=torch.bfloat16).to('cuda'); a=rcm_one(p); del p; gc.collect(); torch.cuda.empty_cache()
    p=WanPipeline.from_pretrained(model,torch_dtype=torch.bfloat16).to('cuda'); helper.load_quantized_transformer(p,DATA/'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16',model); b=rcm_one(p); del p; gc.collect(); torch.cuda.empty_cache()
    return {'settings':{'seed':SEED,'steps':4,'guidance':0,'size':[480,832],'frames':77,'schedule':'rCM TrigFlow->RectifiedFlow'},'outputs':{'velocity':metric(a,b)}}

def main():
    if os.environ.get("ONLY_WAN") == "1":
        report=json.loads(OUT.read_text())
        report["models"]["wan2.1-1.3b"]=wan()
        OUT.write_text(json.dumps(report,indent=2)+"\n")
        return
    report={'definition':{'mse':'mean((quantized-bf16)^2)','nmse':'mse / mean(bf16^2)','method':'existing inference pipelines; transformer final-output hook; stop after first scheduler step'},'models':{}}
    for n,f in [('wan2.1-1.3b',wan),('rcm-wan2.1-1.3b',rcm),('flux.1-dev',lambda:flux('FLUX.1-dev',50,3.5)),('flux.1-schnell',lambda:flux('FLUX.1-schnell',4,0.0))]:
        print('===',n,flush=True); report['models'][n]=f(); OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
