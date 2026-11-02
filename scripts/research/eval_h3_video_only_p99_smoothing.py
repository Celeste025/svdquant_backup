#!/usr/bin/env python3
"""E094: mixed versus video-only p99 SmoothQuant on a real H3 FC2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import KEY, WEIGHT, load_input
from eval_h3_smoothquant_nvfp4_fc2 import ALPHAS, activation_channel_stat


CALIB = Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p11_s00_calib.pt')
EVAL = Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt')
BOUNDARIES = {'calibration': (738, 1152), 'evaluation': (658, 1072)}
VIDEO_ROWS = 21312


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/research/E094/h3_fc2_video_only_p99_nvfp4.json'))
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--smoke-rows', type=int, default=0,
                   help='truncate video rows after p99 statistics; smoke only')
    return p.parse_args()


def nmse(actual, target):
    return float(((actual.float()-target.float()).square().sum()/target.float().square().sum()).item())


def global_inverse(x):
    amax = float(x.abs().amax().item())
    return torch.tensor([2688.0/amax], device=x.device, dtype=torch.float32), amax


def native(x, wq, wsf, xi, wi, nvfp4_quantize, mm_fp4, sf_layout):
    xq, xsf = nvfp4_quantize(x, xi, sfLayout=sf_layout, backend='cuda')
    y = mm_fp4(xq, wq.T, xsf, wsf.T, alpha=(xi*wi).reciprocal(),
               out_dtype=torch.bfloat16, backend='cutlass')
    del xq, xsf
    return y


def main():
    a = arguments()
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    torch.cuda.set_device(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{a.gpu}')
    c_cpu, c_call = load_input(CALIB)
    e_cpu, e_call = load_input(EVAL)
    assert c_call != e_call
    assert tuple(c_cpu.shape) == (22464,14336)
    assert tuple(e_cpu.shape) == (22400,14336)
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as handle:
        w_cpu = handle.get_tensor(KEY)
    assert tuple(w_cpu.shape) == (5376,14336)
    report = dict(format='h3-video-only-p99-smoothing-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='one H3 block-25 FC2, p11 calibration, p1 held-out; video-only output NMSE',
                  calibration=str(CALIB), evaluation=str(EVAL),
                  calibration_call=c_call, evaluation_call=e_call,
                  weight=str(WEIGHT), weight_key=KEY, gpu=a.gpu,
                  smoke_rows=a.smoke_rows, boundaries=BOUNDARIES,
                  alpha_grid=[None,*ALPHAS], modes={})
    with torch.inference_mode():
        c = c_cpu.to(device)
        e = e_cpu.to(device)
        w = w_cpu.to(device)
        del c_cpu, e_cpu, w_cpu
        c_video = c[BOUNDARIES['calibration'][1]:BOUNDARIES['calibration'][1]+VIDEO_ROWS].contiguous()
        # p1 has 16 packed padding rows after its 21,312 actual video rows.
        e_video = e[BOUNDARIES['evaluation'][1]:BOUNDARIES['evaluation'][1]+VIDEO_ROWS].contiguous()
        assert c_video.shape[0] == e_video.shape[0] == VIDEO_ROWS
        assert c_video.shape[1] == e_video.shape[1] == w.shape[1]
        mixed_p99 = activation_channel_stat(c, 'abs-p99')
        video_p99 = activation_channel_stat(c_video, 'abs-p99')
        ratio = video_p99/mixed_p99
        report['p99_comparison'] = dict(
            mixed_vs_video_pearson=float(torch.corrcoef(torch.stack([mixed_p99,video_p99]))[0,1].item()),
            video_to_mixed_ratio_quantiles={str(q):float(torch.quantile(ratio,q).item()) for q in (0,.01,.1,.5,.9,.99,1)},
            channels_change_over_10pct=int(((ratio-1).abs()>.1).sum().item()),
            channels_total=ratio.numel())
        print('p99 comparison',report['p99_comparison'],flush=True)
        if a.smoke_rows:
            assert a.smoke_rows > 0 and a.smoke_rows <= VIDEO_ROWS and a.smoke_rows % 64 == 0
            c_video = c_video[:a.smoke_rows].contiguous()
            e_video = e_video[:a.smoke_rows].contiguous()
        c_ref = c_video @ w.T
        e_ref = e_video @ w.T
        w_stat = w.abs().amax(dim=0).float().clamp_min(1e-5)
        for mode, channel_stat, c_scale_source in (
            ('mixed',mixed_p99,c), ('video_only',video_p99,c_video)):
            rows = []
            for alpha in (None,*ALPHAS):
                if alpha is None:
                    scale = torch.ones_like(w_stat)
                else:
                    scale = (channel_stat.pow(alpha)/w_stat.pow(1-alpha)).clamp_min(1e-5)
                ws = (w.float()*scale).to(torch.bfloat16)
                cs = (c_video.float()/scale).to(torch.bfloat16)
                es = (e_video.float()/scale).to(torch.bfloat16)
                # Mixed calibration uses the whole original sequence for its
                # activation global scale, even though both metrics use video.
                if mode == 'mixed':
                    c_all_s = (c_scale_source.float()/scale).to(torch.bfloat16)
                    ci, c_amax = global_inverse(c_all_s)
                    del c_all_s
                else:
                    ci, c_amax = global_inverse(cs)
                ei, e_amax = global_inverse(es)
                wi, w_amax = global_inverse(ws)
                wq, wsf = nvfp4_quantize(ws, wi, sfLayout=SfLayout.layout_128x4,backend='cuda')
                c_out = native(cs,wq,wsf,ci,wi,nvfp4_quantize,mm_fp4,SfLayout.layout_128x4)
                e_static = native(es,wq,wsf,ci,wi,nvfp4_quantize,mm_fp4,SfLayout.layout_128x4)
                e_dynamic = native(es,wq,wsf,ei,wi,nvfp4_quantize,mm_fp4,SfLayout.layout_128x4)
                row = dict(alpha=alpha, calibration_video_nmse=nmse(c_out,c_ref),
                           evaluation_video_static_nmse=nmse(e_static,e_ref),
                           evaluation_video_dynamic_nmse=nmse(e_dynamic,e_ref),
                           calibration_activation_amax=c_amax,
                           evaluation_video_activation_amax=e_amax,
                           evaluation_to_calibration_amax=e_amax/c_amax,
                           weight_amax=w_amax,
                           scale_min=float(scale.min().item()),
                           scale_median=float(scale.median().item()),
                           scale_max=float(scale.max().item()),
                           bf16_reparameterization_calibration_nmse=nmse(cs@ws.T,c_ref),
                           bf16_reparameterization_evaluation_nmse=nmse(es@ws.T,e_ref))
                rows.append(row)
                print(mode,alpha,row['calibration_video_nmse'],
                      row['evaluation_video_static_nmse'],
                      row['evaluation_video_dynamic_nmse'],flush=True)
                del scale, ws,cs,es,wq,wsf,c_out,e_static,e_dynamic
            selected = min(rows[1:],key=lambda r:r['calibration_video_nmse'])
            report['modes'][mode] = dict(rows=rows,calibration_selected_alpha=selected['alpha'])
        # A linear has no token mixing. This explicit test makes the subset
        # baseline relation independently checkable with the reported values.
        assert abs(report['modes']['mixed']['rows'][0]['evaluation_video_dynamic_nmse']-
                   report['modes']['video_only']['rows'][0]['evaluation_video_dynamic_nmse']) < 1e-8
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('wrote',a.output,flush=True)


if __name__ == '__main__':
    main()
