#!/usr/bin/env python3
"""Teacher-state temporal error diagnostic, not a solver/quality claim.

Uses actual 33-frame Wan BF16 cached trajectory states and both CFG branches.
W4A16 disables only activation Quantizer ProcessHooks, preserving smoothing and
low-rank hooks. The activation increment is W4A4-W4A16, not standalone W16A4.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from infer_rcm_wan_4step import load_quantized_transformer
from diffusers import WanPipeline, WanTransformer3DModel
from deepcompressor.utils.hooks.processor import ProcessHook
from deepcompressor.quantizer.processor import Quantizer

DATA=Path('/data1/models/svdquant-wjq')


def energy(x): return x.double().square().sum().item()


@contextmanager
def activation_bypass(hooks, enabled):
    saved=[h.func for h in hooks]
    try:
        if enabled:
            for h in hooks: h.func=lambda x:x
        yield
    finally:
        for h, f in zip(hooks,saved,strict=True):h.func=f


def matrix_stats(errors):
    flat=torch.stack([x.float().flatten() for x in errors])
    # Float64 dot sums avoid a TF32/default matmul confound in statistics.
    gram=(flat.double()@flat.double().T)
    norms=gram.diag().clamp_min(1e-30).sqrt()
    cosine=gram/(norms[:,None]*norms[None,:])
    diagonal=gram.diag().sum().item()
    return {'cosine':cosine.tolist(), 'gram':gram.tolist(),
            'adjacent_cosine_mean':cosine.diag(1).mean().item(),
            'equal_weight_sum_energy_over_independent':gram.sum().item()/max(diagonal,1e-30),
            'warning':'equal-weight statistic is not actual UniPC solver endpoint error'}


def subtract_channel_mean(x):
    return x-x.mean(dim=tuple(range(2,x.ndim)),keepdim=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',default='0019,0022')
    p.add_argument('--steps',default='22,23,24,25,26,27')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tensor-dir',type=Path,required=True)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.tensor_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False
    model=DATA/'models/Wan2.1-T2V-1.3B-Diffusers'
    ckpt=DATA/'ckpts/wan2.1-1.3b-real-nvfp4-s16'
    cache=DATA/'datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16/caches'
    bf=WanTransformer3DModel.from_pretrained(model/'transformer',torch_dtype=torch.bfloat16).cuda().eval()
    qp=WanPipeline.from_pretrained(model,torch_dtype=torch.bfloat16,text_encoder=None,tokenizer=None,vae=None)
    qp.transformer.cuda().eval()
    load_quantized_transformer(qp,ckpt,model)
    hooks=[];names=[]
    for name,module in qp.transformer.named_modules():
        for h in module._forward_pre_hooks.values():
            if isinstance(h,ProcessHook) and isinstance(h.processor,Quantizer):
                hooks.append(h);names.append(name)
    if len(hooks)!=300:raise RuntimeError(f'Expected 300 input quantizers, found {len(hooks)}')
    print('READY: 300 activation hooks; preserving all other hooks',flush=True)
    steps=[int(s) for s in args.steps.split(',')]
    rows=[];series=[]
    report={'status':'running','model':str(model),'checkpoint':str(ckpt),'cache':str(cache),
            'native_fp4':False,'torch':torch.__version__,'steps':steps,'cfg_scale':6.0,
            'protocol':'BF16 teacher states; no closed-loop quantized rollout',
            'activation_bypass_hooks':names,'rows':rows,'series':series}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for pid in args.prompts.split(','):
        storage={v:[] for v in ['w4a4_cond','w4a16_cond','activation_increment_cond',
                                 'w4a4_cfg','w4a16_cfg','activation_increment_cfg']}
        for step in steps:
            started=time.time(); branch_outputs={}
            for branch in (0,1):
                path=cache/f'{pid}-0-{step:05d}-{branch}.pt'
                payload=torch.load(path,map_location='cpu',weights_only=False)
                x=payload['input_args'][0].cuda()
                kw={k:v.cuda() if torch.is_tensor(v) else v for k,v in payload['input_kwargs'].items()}
                ref=bf(x,**kw)[0].float()
                cache_ref=payload['outputs']
                if isinstance(cache_ref,(tuple,list)):cache_ref=cache_ref[0]
                replay_nmse=energy(ref-cache_ref.cuda().float())/max(energy(ref),1e-30)
                if replay_nmse>1e-8:raise RuntimeError(f'cache mismatch {path}: {replay_nmse}')
                with activation_bypass(hooks,False): wa=qp.transformer(x,**kw)[0].float()
                with activation_bypass(hooks,True): wo=qp.transformer(x,**kw)[0].float()
                branch_outputs[branch]=[t.cpu() for t in (ref,wa,wo)]
                rows.append({'prompt':pid,'step':step,'branch':branch,'timestep':kw['timestep'].tolist(),
                             'shape':list(x.shape),'cache_replay_nmse':replay_nmse,
                             'w4a4_nmse':energy(wa-ref)/energy(ref),
                             'w4a16_nmse':energy(wo-ref)/energy(ref)})
                del x,kw,ref,wa,wo,cache_ref,payload
            cond,uncond=branch_outputs[0],branch_outputs[1]
            cfg=[u+6*(c-u) for c,u in zip(cond,uncond,strict=True)]
            for suffix,outputs in [('cond',cond),('cfg',cfg)]:
                ref,wa,wo=outputs
                storage['w4a4_'+suffix].append(wa-ref)
                storage['w4a16_'+suffix].append(wo-ref)
                storage['activation_increment_'+suffix].append(wa-wo)
            print(json.dumps({'prompt':pid,'step':step,'rows':rows[-2:],'seconds':time.time()-started}),flush=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        tensor_path=args.tensor_dir/f'{pid}_errors.pt'
        torch.save({'errors':storage,'steps':steps,'prompt':pid,'cfg_scale':6.0},tensor_path)
        for variant,errors in storage.items():
            rec={'prompt':pid,'variant':variant,'raw':matrix_stats(errors),
                 'channel_centered':matrix_stats([subtract_channel_mean(x) for x in errors])}
            series.append(rec)
            print(json.dumps({'prompt':pid,'variant':variant,
                              'raw_adj_cos':rec['raw']['adjacent_cosine_mean'],
                              'centered_adj_cos':rec['channel_centered']['adjacent_cosine_mean']}),flush=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        del storage
    report['status']='complete'; report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
