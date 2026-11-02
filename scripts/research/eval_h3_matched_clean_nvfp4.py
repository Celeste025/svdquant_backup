#!/usr/bin/env python3
"""Paired plain versus p99-smoothed W4A4 on identical cleaned H3 video input."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import KEY, WEIGHT, load_input
from eval_h3_smoothquant_nvfp4_fc2 import activation_channel_stat
from eval_h3_video_only_p99_smoothing import CALIB, EVAL, VIDEO_ROWS


OUT = Path('results/research/E095/h3_fc2_matched_clean_video_nvfp4.json')


def nmse(actual, target):
    return float(((actual.float()-target.float()).square().sum()/target.float().square().sum()).item())


def main():
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    torch.cuda.set_device(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device('cuda:6')
    c_cpu, c_call = load_input(CALIB)
    e_cpu, e_call = load_input(EVAL)
    assert c_call != e_call
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(KEY)
    with torch.inference_mode():
        c = c_cpu[1152:1152+VIDEO_ROWS].to(dev)
        e = e_cpu[1072:1072+VIDEO_ROWS].to(dev)
        w = w_cpu.to(dev)
        del c_cpu, e_cpu, w_cpu
        assert c.shape == e.shape == (VIDEO_ROWS,14336)
        alpha = .5
        p99 = activation_channel_stat(c,'abs-p99')
        wmax = w.abs().amax(dim=0).float().clamp_min(1e-5)
        scale = (p99.pow(alpha)/wmax.pow(1-alpha)).clamp_min(1e-5)
        cs = (c.float()/scale).to(torch.bfloat16)
        es = (e.float()/scale).to(torch.bfloat16)
        plain_calib_amax = float(c.abs().amax().item())
        smooth_calib_amax = float(cs.abs().amax().item())
        assert plain_calib_amax == 2336.0 and smooth_calib_amax == 388.0
        masks = {
            'none': torch.zeros_like(e,dtype=torch.bool),
            'raw_above_p11_video_amax': e.abs()>plain_calib_amax,
            'smoothed_above_p11_video_amax': es.abs()>smooth_calib_amax,
        }
        original_target=e@w.T
        original_energy=original_target.float().square().sum()
        rows=[]
        for mask_name,mask in masks.items():
            cleaned=e.masked_fill(mask,0)
            target=cleaned@w.T
            arms={}
            for arm,x,ww,calib_amax in (
                ('plain',cleaned,w,plain_calib_amax),
                ('p99_smooth_alpha_0.5',(cleaned.float()/scale).to(torch.bfloat16),
                 (w.float()*scale).to(torch.bfloat16),smooth_calib_amax),
            ):
                wi=torch.tensor([2688.0/float(ww.abs().amax().item())],device=dev,dtype=torch.float32)
                wq,wsf=nvfp4_quantize(ww,wi,sfLayout=SfLayout.layout_128x4,backend='cuda')
                results={}
                for policy,amax in (('static_p11_video',calib_amax),
                                    ('current_input',float(x.abs().amax().item()))):
                    xi=torch.tensor([2688.0/amax],device=dev,dtype=torch.float32)
                    aq,asf=nvfp4_quantize(x,xi,sfLayout=SfLayout.layout_128x4,backend='cuda')
                    y=mm_fp4(aq,wq.T,asf,wsf.T,alpha=(xi*wi).reciprocal(),
                             out_dtype=torch.bfloat16,backend='cutlass')
                    results[policy]=nmse(y,target)
                    del aq,asf,y
                arms[arm]=dict(activation_amax=float(x.abs().amax().item()),
                               bf16_reparameterization_nmse=0.0 if arm=='plain' else nmse(x@ww.T,target),
                               nmse=results)
                del wq,wsf
            row=dict(mask=mask_name,removed_elements=int(mask.sum().item()),
                     removed_fraction=float(mask.float().mean().item()),
                     cleaned_target_energy_ratio=float(target.float().square().sum().item()/
                                                       original_energy.item()),
                     arms=arms)
            rows.append(row)
            print(mask_name,row,flush=True)
            del cleaned,target
    report=dict(format='h3-fc2-matched-clean-video-nvfp4-v1',
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scope='p1 video rows, identical zero mask and cleaned BF16 target for plain versus p11-video-p99 alpha .5 smoothing',
                calibration=str(CALIB),evaluation=str(EVAL),
                calibration_call=c_call,evaluation_call=e_call,
                plain_calibration_amax=plain_calib_amax,
                smooth_calibration_amax=smooth_calib_amax,
                rows=rows)
    # No-zero rows must reproduce E094's video-only baseline on both arms.
    baseline=rows[0]['arms']
    assert abs(baseline['plain']['nmse']['static_p11_video']-.010277296416461468)<2e-7
    assert abs(baseline['plain']['nmse']['current_input']-.00968239177018404)<2e-7
    assert abs(baseline['p99_smooth_alpha_0.5']['nmse']['static_p11_video']-.010862885974347591)<2e-7
    assert abs(baseline['p99_smooth_alpha_0.5']['nmse']['current_input']-.009722449816763401)<2e-7
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('wrote',OUT,flush=True)


if __name__=='__main__':
    main()
