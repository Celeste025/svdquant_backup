#!/usr/bin/env python3
"""Single-layer W4A4 intervention: do scale-switch effects survive the full DiT?

All untargeted modules stay BF16. We freeze the saved SVDQuant residual weights
and rank-32 factors. Pair inputs share timestep/text; the latent perturbation
uses a real next-step direction rescaled to 1% or 10% relative RMS.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from exp_nvfp4_scale_jitter import CACHE, CKPT, MODEL, scale_for, qdq, write_csv


def sums(e): return e.float().square().sum().item()


@torch.inference_mode()
def activation_pairs(args,model,weights,smooth,branches):
    rows=[]
    names=['blocks.20.attn1.to_q','blocks.20.ffn.net.2']
    for pid in args.prompts.split(','):
        for step in [0,2]:
            inputs={};snapshots=[];block={}
            def capture_input(name):
                def hook(m,a):inputs[name]=a[0].detach().clone()
                return hook
            handles=[model.get_submodule(n).register_forward_pre_hook(capture_input(n)) for n in names]
            def bhook(m,a,out):block['value']=out.detach().clone()
            handle=model.blocks[20].register_forward_hook(bhook)
            ps=[torch.load(CACHE/f'{pid}-{s:05d}-0.pt',map_location='cpu',weights_only=False) for s in [step,step+1]]
            for i,p in enumerate(ps):
                kw={k:v.cuda() if torch.is_tensor(v) else v for k,v in p['input_kwargs'].items()}
                out=model(p['input_args'][0].cuda(),**kw)[0].float()
                if i==0:ref0=out;bref0=block['value'].float()
                snapshots.append(inputs.copy())
            for h in handles:h.remove()
            p=ps[0];root_input=p['input_args'][0].cuda()
            kw={k:v.cuda() if torch.is_tensor(v) else v for k,v in p['input_kwargs'].items()}
            for name in names:
                mod=model.get_submodule(name);original=mod.forward
                scale=smooth[name].cuda().to(torch.bfloat16)
                z0=(snapshots[0][name]/scale).to(torch.bfloat16)
                dz=(snapshots[1][name]/scale).float()-z0.float()
                rel=(dz.square().mean()/z0.float().square().mean()).sqrt()
                intervention=Intervention(name,weights[name+'.weight'],smooth[name],branches[name],weights.get(name+'.bias'))
                mod.forward=intervention
                out0=model(root_input,**kw)[0].float();bout0=block['value'].float()
                base_metrics=intervention.metrics.copy();mod.forward=original
                for eps in [.01,.1]:
                    replacement=((z0.float()+dz*(eps/rel))*scale.float()).to(torch.bfloat16)
                    def replace(m,a):return (replacement,*a[1:])
                    h=mod.register_forward_pre_hook(replace)
                    try:
                        ref1=model(root_input,**kw)[0].float();bref1=block['value'].float()
                        mod.forward=intervention
                        for mode in ['dynamic_global','fixed_global','frozen_block','hysteresis_mse5']:
                            intervention.mode=mode
                            out1=model(root_input,**kw)[0].float();bout1=block['value'].float()
                            row={'prompt':pid,'step':step,'eps':eps,'layer':name,'mode':mode,
                                 'perturb_site':'activation',**intervention.metrics}
                            for prefix,a,b,ra,rb in [('final',out0,out1,ref0,ref1),('block',bout0,bout1,bref0,bref1)]:
                                row[prefix+'_response_err2']=sums((b-a)-(rb-ra))
                                row[prefix+'_response_ref2']=sums(rb-ra)
                                row[prefix+'_point_err2']=sums(a-ra)+sums(b-rb)
                                row[prefix+'_point_ref2']=sums(ra)+sums(rb)
                                row[prefix+'_response_nmse']=row[prefix+'_response_err2']/max(row[prefix+'_response_ref2'],1e-30)
                                row[prefix+'_point_nmse']=row[prefix+'_point_err2']/max(row[prefix+'_point_ref2'],1e-30)
                            row['a_base_err2']=base_metrics['a_err2'];row['a_base_ref2']=base_metrics['a_ref2']
                            rows.append(row)
                            print(json.dumps({k:row[k] for k in ['prompt','step','eps','layer','mode','final_response_nmse','final_point_nmse']}),flush=True)
                            write_csv(args.output/f'functional_activation_{args.prompts.replace(",","_")}.csv',rows)
                    finally:
                        mod.forward=original;h.remove()
                del intervention
            handle.remove()


class Intervention:
    def __init__(self,name,weight,smooth,branch,bias):
        self.name=name;self.weight=weight.cuda();self.smooth=smooth.cuda().to(torch.bfloat16)
        self.a=branch['a.weight'].cuda();self.b=branch['b.weight'][:weight.shape[0]].cuda()
        self.bias=bias.cuda() if bias is not None else None
        self.mode='base';self.s0=None;self.x0=None;self.q0=None;self.g0=None;self.metrics={}

    @torch.inference_mode()
    def __call__(self,x):
        original_shape=x.shape
        sm=(x.reshape(-1,x.shape[-1])/self.smooth).to(torch.bfloat16)
        g=sm.float().abs().amax().clamp_min(1e-12)/(6*448)
        if self.mode=='base':
            self.g0=g;self.s0=[];self.x0=sm.clone();self.q0=[]
        effective_g=g if self.mode in ['base','dynamic_global'] else self.g0
        output=torch.empty((*sm.shape[:-1],self.weight.shape[0]),device=x.device,dtype=x.dtype)
        acc={'a_err2':0.,'a_ref2':0.,'a_response_err2':0.,'a_response_ref2':0.,
             'switched':0.,'groups':0.,'overflow':0.,'values':0.}
        for start in range(0,sm.shape[0],256):
            stop=min(start+256,sm.shape[0]); xx=sm[start:stop].float().reshape(-1,sm.shape[-1]//16,16)
            ss=scale_for(xx,effective_g);qq=qdq(xx,ss)
            if self.mode=='base':
                self.s0.append(ss);self.q0.append(qq.to(torch.bfloat16))
            else:
                old_s=self.s0[start:stop]
                if self.mode=='frozen_block':ss=old_s;qq=qdq(xx,ss)
                elif self.mode=='hysteresis_mse5':
                    qf=qdq(xx,old_s)
                    keep=(qf-xx).square().sum(-1,keepdim=True)<=1.05*(qq-xx).square().sum(-1,keepdim=True)
                    ss=torch.where(keep,old_s,ss);qq=torch.where(keep,qf,qq)
                old_x=self.x0[start:stop].float().reshape_as(xx)
                old_q=self.q0[start:stop].float()
                acc['a_response_err2']+=sums((qq.to(torch.bfloat16).float()-old_q)-(xx-old_x))
                acc['a_response_ref2']+=sums(xx-old_x)
                acc['switched']+=((ss-old_s).abs()>old_s.abs()*1e-6).sum().item()
                acc['groups']+=ss.numel()
            acc['a_err2']+=sums(qq.to(torch.bfloat16).float()-xx);acc['a_ref2']+=sums(xx)
            acc['overflow']+=(xx.abs()>6*ss).sum().item();acc['values']+=xx.numel()
            # The FP16/BF16 low-rank branch receives unquantized smoothed inputs.
            qflat=qq.reshape(stop-start,-1).to(torch.bfloat16)
            out=F.linear(qflat,self.weight,self.bias)
            out.add_(F.linear(F.linear(sm[start:stop],self.a),self.b))
            output[start:stop]=out
        if self.mode=='base':
            self.s0=torch.cat(self.s0);self.q0=torch.cat(self.q0)
        acc['global_scale']=g.item();acc['effective_global_scale']=effective_g.item()
        self.metrics=acc
        return output.reshape(*original_shape[:-1],self.weight.shape[0])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prompts',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--perturb-site',choices=['latent','activation'],default='latent')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6)
    from diffusers import WanTransformer3DModel
    model=WanTransformer3DModel.from_pretrained(MODEL,torch_dtype=torch.bfloat16).cuda().eval()
    weights=torch.load(CKPT/'model.pt',map_location='cpu',weights_only=False)
    smooth=torch.load(CKPT/'smooth.pt',map_location='cpu',weights_only=False)
    branches=torch.load(CKPT/'branch.pt',map_location='cpu',weights_only=False)
    if args.perturb_site=='activation':
        activation_pairs(args,model,weights,smooth,branches)
        return
    rows=[]
    for pid in args.prompts.split(','):
        for step in [0,2]:
            p=torch.load(CACHE/f'{pid}-{step:05d}-0.pt',map_location='cpu',weights_only=False)
            pn=torch.load(CACHE/f'{pid}-{step+1:05d}-0.pt',map_location='cpu',weights_only=False)
            kw={k:v.cuda() if torch.is_tensor(v) else v for k,v in p['input_kwargs'].items()}
            x0=p['input_args'][0].cuda();delta=pn['input_args'][0].cuda().float()-x0.float()
            norm=(delta.square().mean()/x0.float().square().mean()).sqrt()
            block={}
            def block_hook(m,a,out):block['value']=out.detach().clone()
            handle=model.blocks[20].register_forward_hook(block_hook)
            ref0=model(x0,**kw)[0].float();bref0=block['value'].float()
            for eps in [.01,.1]:
                x1=(x0.float()+delta*(eps/norm)).to(torch.bfloat16)
                ref1=model(x1,**kw)[0].float();bref1=block['value'].float()
                for name in ['blocks.20.attn1.to_q','blocks.20.ffn.net.2']:
                    mod=model.get_submodule(name);original=mod.forward
                    intervention=Intervention(name,weights[name+'.weight'],smooth[name],branches[name],weights.get(name+'.bias'))
                    mod.forward=intervention
                    started=time.time()
                    try:
                        out0=model(x0,**kw)[0].float();bout0=block['value'].float()
                        baseline_metrics=intervention.metrics.copy()
                        for mode in ['dynamic_global','fixed_global','frozen_block','hysteresis_mse5']:
                            intervention.mode=mode
                            out1=model(x1,**kw)[0].float();bout1=block['value'].float()
                            row={'prompt':pid,'step':step,'eps':eps,'layer':name,'mode':mode,
                                 'latent_relative_delta':(sums(x1.float()-x0.float())/sums(x0)).__pow__(.5),
                                 **intervention.metrics}
                            for prefix,a,b,ra,rb in [('final',out0,out1,ref0,ref1),('block',bout0,bout1,bref0,bref1)]:
                                row[prefix+'_response_err2']=sums((b-a)-(rb-ra))
                                row[prefix+'_response_ref2']=sums(rb-ra)
                                row[prefix+'_point_err2']=sums(a-ra)+sums(b-rb)
                                row[prefix+'_point_ref2']=sums(ra)+sums(rb)
                                row[prefix+'_response_nmse']=row[prefix+'_response_err2']/max(row[prefix+'_response_ref2'],1e-30)
                                row[prefix+'_point_nmse']=row[prefix+'_point_err2']/max(row[prefix+'_point_ref2'],1e-30)
                            row['a_base_err2']=baseline_metrics['a_err2'];row['a_base_ref2']=baseline_metrics['a_ref2']
                            rows.append(row)
                            print(json.dumps({k:row[k] for k in ['prompt','step','eps','layer','mode','final_response_nmse','final_point_nmse']}),flush=True)
                            write_csv(args.output/f'functional_{args.prompts.replace(",","_")}.csv',rows)
                    finally:
                        mod.forward=original
                    print(f'case completed in {time.time()-started:.1f}s',flush=True)
                    del intervention
            handle.remove()


if __name__=='__main__':main()
