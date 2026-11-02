#!/usr/bin/env python3
"""E096: exact top-0.1% raw-activation clearing, paired H3 FC2 W4A4 test."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import KEY, WEIGHT, load_input
from eval_h3_smoothquant_nvfp4_fc2 import activation_channel_stat
from eval_h3_video_only_p99_smoothing import CALIB, EVAL, VIDEO_ROWS


OUT = Path('results/research/E096/h3_fc2_top01pct_clean_video_nvfp4.json')
FRACTION = 0.001


def nmse(a, b):
    return float(((a.float()-b.float()).square().sum()/b.float().square().sum()).item())


def top_mask(x):
    flat = x.abs().flatten()
    k = math.ceil(FRACTION*flat.numel())
    values = torch.topk(flat, k, sorted=False).values
    cutoff = values.min()
    strict = flat > cutoff
    strict_count = int(strict.sum().item())
    ties_needed = k-strict_count
    tie_indices = torch.nonzero(flat == cutoff, as_tuple=False).flatten()[:ties_needed]
    mask = strict
    mask[tie_indices] = True
    assert int(mask.sum().item()) == k
    return mask.view_as(x), dict(k=k,total=flat.numel(),cutoff=float(cutoff.item()),
                                 strictly_greater=strict_count,
                                 ties_selected=ties_needed)


def evaluate_arm(c, e, w, c_ref, e_ref, channel_scale,
                 nvfp4_quantize, mm_fp4, sf_layout):
    if channel_scale is None:
        cs,es,ws = c,e,w
        reparam = dict(calibration=0.0,evaluation=0.0)
    else:
        cs=(c.float()/channel_scale).to(torch.bfloat16)
        es=(e.float()/channel_scale).to(torch.bfloat16)
        ws=(w.float()*channel_scale).to(torch.bfloat16)
        reparam=dict(calibration=nmse(cs@ws.T,c_ref),
                     evaluation=nmse(es@ws.T,e_ref))
    w_amax=float(ws.abs().amax().item())
    wi=torch.tensor([2688.0/w_amax],device=w.device,dtype=torch.float32)
    wq,wsf=nvfp4_quantize(ws,wi,sfLayout=sf_layout,backend='cuda')
    c_amax=float(cs.abs().amax().item())
    e_amax=float(es.abs().amax().item())
    ci=torch.tensor([2688.0/c_amax],device=w.device,dtype=torch.float32)
    ei=torch.tensor([2688.0/e_amax],device=w.device,dtype=torch.float32)
    outputs={}
    for label,x,target in (('calibration',cs,c_ref),('evaluation',es,e_ref)):
        policies={}
        for policy,xi in (('static_p11',ci),('current_input',ci if label=='calibration' else ei)):
            aq,asf=nvfp4_quantize(x,xi,sfLayout=sf_layout,backend='cuda')
            y=mm_fp4(aq,wq.T,asf,wsf.T,alpha=(xi*wi).reciprocal(),
                     out_dtype=torch.bfloat16,backend='cutlass')
            policies[policy]=nmse(y,target)
            del aq,asf,y
        outputs[label]=policies
    return dict(output_nmse=outputs,bf16_reparameterization_nmse=reparam,
                activation_amax=dict(calibration=c_amax,evaluation=e_amax),
                weight_amax=w_amax)


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
    report=dict(format='h3-top-0.1pct-clean-video-nvfp4-v1',
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                scope='block-25 FC2; p11 calibration, p1 held-out; 21,312 video rows each',
                calibration=str(CALIB),evaluation=str(EVAL),
                calibration_call=c_call,evaluation_call=e_call,
                fraction=FRACTION,alpha=.5,
                mask_rule='exact largest ceil(0.001*N) raw |X| entries within each video tensor; ties broken by lowest flattened index',
                target='the same cleaned BF16 XW.T for plain and smoothed W4A4',
                cases={})
    with torch.inference_mode():
        c=c_cpu[1152:1152+VIDEO_ROWS].to(dev)
        e=e_cpu[1072:1072+VIDEO_ROWS].to(dev)
        w=w_cpu.to(dev)
        del c_cpu,e_cpu,w_cpu
        assert c.shape==e.shape==(VIDEO_ROWS,14336)
        c_mask,c_mask_info=top_mask(c)
        e_mask,e_mask_info=top_mask(e)
        c_clean=c.masked_fill(c_mask,0)
        e_clean=e.masked_fill(e_mask,0)
        original_refs=dict(calibration=c@w.T,evaluation=e@w.T)
        for label,cc,ee in (('none',c,e),('top_0.1pct_zero',c_clean,e_clean)):
            c_ref=original_refs['calibration'] if label=='none' else cc@w.T
            e_ref=original_refs['evaluation'] if label=='none' else ee@w.T
            stat=activation_channel_stat(cc,'abs-p99')
            wstat=w.abs().amax(dim=0).float().clamp_min(1e-5)
            scale=(stat.sqrt()/wstat.sqrt()).clamp_min(1e-5)
            arms={
                'plain':evaluate_arm(cc,ee,w,c_ref,e_ref,None,nvfp4_quantize,mm_fp4,SfLayout.layout_128x4),
                'p99_smooth_alpha_0.5':evaluate_arm(cc,ee,w,c_ref,e_ref,scale,nvfp4_quantize,mm_fp4,SfLayout.layout_128x4),
            }
            report['cases'][label]=dict(
                masks=dict(calibration={'k':0} if label=='none' else c_mask_info,
                           evaluation={'k':0} if label=='none' else e_mask_info),
                original_to_clean_bf16_nmse=dict(
                    calibration=nmse(c_ref,original_refs['calibration']),
                    evaluation=nmse(e_ref,original_refs['evaluation'])),
                clean_to_original_output_energy_ratio=dict(
                    calibration=float((c_ref.float().square().sum()/original_refs['calibration'].float().square().sum()).item()),
                    evaluation=float((e_ref.float().square().sum()/original_refs['evaluation'].float().square().sum()).item())),
                smoothing_scale_stats=dict(min=float(scale.min().item()),
                                           median=float(scale.median().item()),
                                           max=float(scale.max().item())),
                arms=arms)
            print(label,report['cases'][label],flush=True)
        # Reproduce E094's no-clean p11-video p99 alpha-0.5 and plain p1 baselines.
        n=report['cases']['none']['arms']
        assert abs(n['plain']['output_nmse']['evaluation']['current_input']-.00968239177018404)<2e-7
        assert abs(n['p99_smooth_alpha_0.5']['output_nmse']['calibration']['static_p11']-.007276088930666447)<2e-7
        assert abs(n['p99_smooth_alpha_0.5']['output_nmse']['evaluation']['current_input']-.009722449816763401)<2e-7
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print('wrote',OUT,flush=True)


if __name__=='__main__':
    main()
