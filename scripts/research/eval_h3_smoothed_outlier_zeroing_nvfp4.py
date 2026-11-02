#!/usr/bin/env python3
"""E095 targeted check: zero p1 video outliers after p11-video p99 smoothing."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import KEY, WEIGHT, load_input
from eval_h3_smoothquant_nvfp4_fc2 import activation_channel_stat
from eval_h3_video_only_p99_smoothing import CALIB, EVAL, VIDEO_ROWS


OUTPUT = Path('results/research/E095/h3_fc2_smoothed_outlier_zeroing_nvfp4.json')


def nmse(a,b):
    return float(((a.float()-b.float()).square().sum()/b.float().square().sum()).item())


def main():
    from flashinfer import SfLayout,mm_fp4,nvfp4_quantize
    torch.cuda.set_device(6)
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device('cuda:6')
    c_cpu,c_call=load_input(CALIB)
    e_cpu,e_call=load_input(EVAL)
    assert c_call!=e_call
    with safe_open(str(WEIGHT),framework='pt',device='cpu') as f:
        w_cpu=f.get_tensor(KEY)
    with torch.inference_mode():
        c=c_cpu[1152:1152+VIDEO_ROWS].to(dev)
        e=e_cpu[1072:1072+VIDEO_ROWS].to(dev)
        w=w_cpu.to(dev)
        del c_cpu,e_cpu,w_cpu
        assert c.shape==e.shape==(VIDEO_ROWS,14336)
        alpha=.5
        cstat=activation_channel_stat(c,'abs-p99')
        wstat=w.abs().amax(dim=0).float().clamp_min(1e-5)
        scale=(cstat.pow(alpha)/wstat.pow(1-alpha)).clamp_min(1e-5)
        cs=(c.float()/scale).to(torch.bfloat16)
        es=(e.float()/scale).to(torch.bfloat16)
        ws=(w.float()*scale).to(torch.bfloat16)
        ref=e@w.T
        c_amax=float(cs.abs().amax().item())
        e_amax=float(es.abs().amax().item())
        assert c_amax==388.0 and e_amax==900.0
        mask=es.abs()>c_amax
        es0=es.masked_fill(mask,0)
        modified=es0@ws.T
        wi=torch.tensor([2688.0/float(ws.abs().amax().item())],device=dev,dtype=torch.float32)
        ci=torch.tensor([2688.0/c_amax],device=dev,dtype=torch.float32)
        ei=torch.tensor([2688.0/e_amax],device=dev,dtype=torch.float32)
        zi=torch.tensor([2688.0/float(es0.abs().amax().item())],device=dev,dtype=torch.float32)
        wq,wsf=nvfp4_quantize(ws,wi,sfLayout=SfLayout.layout_128x4,backend='cuda')
        rows=[]
        for label,x,target in (('none',es,ref),('zero_above_p11_video_amax',es0,modified)):
            values={}
            for policy,xi in (('static_p11_video',ci),('current_input',ei if label=='none' else zi)):
                aq,asf=nvfp4_quantize(x,xi,sfLayout=SfLayout.layout_128x4,backend='cuda')
                y=mm_fp4(aq,wq.T,asf,wsf.T,alpha=(xi*wi).reciprocal(),
                         out_dtype=torch.bfloat16,backend='cutlass')
                values[policy]=dict(final_original_target_nmse=nmse(y,ref),
                                    quantization_only_modified_target_nmse=nmse(y,target),
                                    quantization_only_original_energy_nmse=float(
                                        ((y.float()-target.float()).square().sum()/ref.float().square().sum()).item()))
                del aq,asf,y
            rows.append(dict(label=label,activation_amax=float(x.abs().amax().item()),
                             removed_elements=0 if label=='none' else int(mask.sum().item()),
                             removed_fraction=0 if label=='none' else float(mask.float().mean().item()),
                             removed_by_video_token_rows=0 if label=='none' else int(mask.any(dim=1).sum().item()),
                             signal_removal_original_target_nmse=nmse(target,ref),
                             values=values))
            print(rows[-1],flush=True)
    report=dict(format='h3-smoothed-outlier-zeroing-nvfp4-v1',
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scope='E094 video-only p99 alpha 0.5 selected on p11; p1 video test only',
                calibration=str(CALIB),evaluation=str(EVAL),
                calibration_call=c_call,evaluation_call=e_call,
                c_amax=c_amax,e_amax=e_amax,alpha=.5,rows=rows)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print('wrote',OUTPUT,flush=True)


if __name__=='__main__':
    main()
