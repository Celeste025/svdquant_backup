#!/usr/bin/env python3
"""Real adjacent-step, single-layer interventions on the 50-step Wan teacher.

Uses cache branch index 0 only; no claim about CFG-combined quality or rollout.
"""
import argparse
import json
from pathlib import Path
import torch
from exp_nvfp4_scale_jitter import DATA, write_csv
from exp_nvfp4_scale_jitter_functional import Intervention, sums


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounding',choices=['rne','deepcompressor'],default='rne')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import exp_nvfp4_scale_jitter
    exp_nvfp4_scale_jitter.ROUNDING_MODE=args.rounding
    torch.set_num_threads(6)
    from diffusers import WanTransformer3DModel
    model=WanTransformer3DModel.from_pretrained(DATA/'models/Wan2.1-T2V-1.3B-Diffusers/transformer',torch_dtype=torch.bfloat16).cuda().eval()
    ckpt=DATA/'ckpts/wan2.1-1.3b-real-nvfp4-s16'
    weights=torch.load(ckpt/'model.pt',map_location='cpu',weights_only=False)
    smooth=torch.load(ckpt/'smooth.pt',map_location='cpu',weights_only=False)
    branches=torch.load(ckpt/'branch.pt',map_location='cpu',weights_only=False)
    cache=DATA/'datasets/torch.bfloat16/wan2.1-1.3b/unipc50-g6.0-f33/vbench/s16/caches'
    rows=[];block={}
    def bhook(m,a,out):block['value']=out.detach().clone()
    handle=model.blocks[20].register_forward_hook(bhook)
    for pid in ['0019','0022']:
        for step in [0,24,48]:
            ps=[torch.load(cache/f'{pid}-0-{s:05d}-0.pt',map_location='cpu',weights_only=False) for s in [step,step+1]]
            def run(i):
                p=ps[i];kw={k:v.cuda() if torch.is_tensor(v) else v for k,v in p['input_kwargs'].items()}
                return model(p['input_args'][0].cuda(),**kw)[0].float()
            ref0=run(0);bref0=block['value'].float()
            ref1=run(1);bref1=block['value'].float()
            for name in ['blocks.20.attn1.to_q','blocks.20.ffn.net.2']:
                mod=model.get_submodule(name);original=mod.forward
                intervention=Intervention(name,weights[name+'.weight'],smooth[name],branches[name],weights.get(name+'.bias'))
                mod.forward=intervention
                try:
                    out0=run(0);bout0=block['value'].float();base=intervention.metrics.copy()
                    for mode in ['dynamic_global','fixed_global','frozen_block','hysteresis_mse5']:
                        intervention.mode=mode
                        out1=run(1);bout1=block['value'].float()
                        row={'prompt':pid,'step':step,'layer':name,'mode':mode,**intervention.metrics,
                             'a_base_err2':base['a_err2'],'a_base_ref2':base['a_ref2']}
                        for prefix,a,b,ra,rb in [('final',out0,out1,ref0,ref1),('block',bout0,bout1,bref0,bref1)]:
                            row[prefix+'_response_err2']=sums((b-a)-(rb-ra))
                            row[prefix+'_response_ref2']=sums(rb-ra)
                            row[prefix+'_point_err2']=sums(a-ra)+sums(b-rb)
                            row[prefix+'_point_ref2']=sums(ra)+sums(rb)
                            row[prefix+'_response_nmse']=row[prefix+'_response_err2']/max(row[prefix+'_response_ref2'],1e-30)
                            row[prefix+'_point_nmse']=row[prefix+'_point_err2']/max(row[prefix+'_point_ref2'],1e-30)
                        prefix='' if args.rounding=='rne' else 'dc_'
                        rows.append(row);write_csv(args.output/f'{prefix}functional_wan_endpoints.csv',rows)
                        print(json.dumps({k:row[k] for k in ['prompt','step','layer','mode','final_response_nmse','final_point_nmse']}),flush=True)
                finally:mod.forward=original
                del intervention
    handle.remove()


if __name__=='__main__':main()
