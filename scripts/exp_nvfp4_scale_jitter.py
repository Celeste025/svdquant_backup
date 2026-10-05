#!/usr/bin/env python3
"""Controlled NVFP4 scale-switch audit on real rCM-Wan activations.

Capture uses unmodified BF16 teacher trajectories. Analysis is numerical Q/DQ,
not a throughput benchmark. Fixed-global probes isolate E4M3 block-scale
switches; endpoint probes also use each full activation tensor's global scale.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DATA = Path('/data1/models/svdquant-wjq')
CACHE = DATA / 'datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches'
MODEL = DATA / 'models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer'
CKPT = DATA / 'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16'
SUFFIXES = ['attn1.to_q', 'attn1.to_out.0', 'ffn.net.0.proj', 'ffn.net.2']
LAYERS = [f'blocks.{b}.{s}' for b in [0, 10, 20, 29] for s in SUFFIXES]
ROUNDING_MODE = 'rne'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader(); w.writerows(rows)


@torch.inference_mode()
def capture(args):
    from diffusers import WanTransformer3DModel
    torch.set_num_threads(6)
    model = WanTransformer3DModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()
    scales = torch.load(args.checkpoint / 'smooth.pt', map_location='cpu', weights_only=False)
    storage = {}
    frame_count = 20
    handles = []
    for name in LAYERS:
        smooth = scales[name].cuda().to(torch.bfloat16)
        def hook(module, inputs, name=name, smooth=smooth):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            assert x.shape[0] % frame_count == 0, x.shape
            spatial_count=x.shape[0]//frame_count
            spatial = torch.linspace(0, spatial_count - 1, 16, device=x.device).long()
            index = (torch.arange(frame_count, device=x.device)[:, None] * spatial_count + spatial).flatten()
            # Match the BF16 smoothing arithmetic used by the runtime.
            sample = (x[index] / smooth).to(torch.bfloat16).cpu()
            amax = torch.zeros((), device=x.device)
            for start in range(0, x.shape[0], 512):
                amax = torch.maximum(amax, (x[start:start+512] / smooth).float().abs().amax())
            storage[name] = {'x': sample, 'full_absmax': float(amax), 'total_tokens': x.shape[0]}
        handles.append(model.get_submodule(name).register_forward_pre_hook(hook))
    verification = []
    for pid in args.prompts.split(','):
        for step in args.steps:
            target = args.output / f'{pid}_s{step}.pt'
            if target.exists():
                print(f'SKIP {target}', flush=True); continue
            filename=f'{pid}-0-{step:05d}-0.pt' if args.standard_wan else f'{pid}-{step:05d}-0.pt'
            p = torch.load(args.cache_dir / filename, map_location='cpu', weights_only=False)
            frame_count=p['input_args'][0].shape[2]
            kw = p['input_kwargs']
            start = time.time()
            storage.clear()
            out = model(hidden_states=p['input_args'][0].cuda(),
                        timestep=kw['timestep'].cuda(),
                        encoder_hidden_states=kw['encoder_hidden_states'].cuda(), return_dict=False)[0]
            ref = p['outputs'][0] if isinstance(p['outputs'], (tuple, list)) else p['outputs']
            err = (out.float() - ref.cuda().float()).square().sum().item()
            power = ref.float().square().sum().item()
            row = {'prompt': pid, 'step': step, 'cache_output_nmse': err / max(power, 1e-30),
                   'seconds': time.time()-start}
            verification.append(row)
            torch.save({'prompt': pid, 'step': step, 'layers': storage.copy(), 'verification': row,
                        'tensor_shape': list(p['input_args'][0].shape), 'timestep': kw['timestep'].tolist()}, target)
            print(json.dumps(row), flush=True)
    (args.output / f'capture_{args.prompts.replace(",", "_")}.json').write_text(json.dumps(verification, indent=2))


def fp4(z):
    # Round to nearest, ties to even E2M1 significand.
    levels = z.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    bounds = z.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5])
    az = z.abs().contiguous()
    idx = torch.bucketize(az, bounds)
    choose_upper = (idx.remainder(2) == 1) if ROUNDING_MODE == 'rne' else (z >= 0)
    tie = (idx < 7) & (az == bounds[idx.clamp_max(6)]) & choose_upper
    idx = idx + tie.long()
    return levels[idx] * z.sign()


def scale_for(x, global_scale, continuous=False):
    ideal = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6
    if continuous:
        return ideal
    raw=(ideal/global_scale).clamp(max=448)
    rounded=raw.to(torch.float8_e4m3fn)
    value=rounded.float()
    if ROUNDING_MODE == 'deepcompressor':
        next_value=(rounded.view(torch.uint8)+1).clamp(max=126).view(torch.float8_e4m3fn).float()
        value=torch.where(raw == (value+next_value)*.5,next_value,value)
    return (value * global_scale).clamp_min(1e-12)


def qdq(x, scale):
    return fp4(x / scale) * scale


def stats(x0, x1, y0, y1, s0, s1, **meta):
    d = x1-x0; e0 = y0-x0; e1 = y1-x1; de = e1-e0
    err0, err1 = e0.square().sum().item(), e1.square().sum().item()
    response_err = de.square().sum().item(); dp = d.square().sum().item()
    switch = ((s1-s0).abs() > s0.abs()*1e-6).squeeze(-1)
    block_err = de.square().sum(-1)
    saturated = (x1.abs() > 6*s1)
    return {**meta, 'response_err2': response_err, 'response_ref2': dp,
            'response_nmse': response_err/max(dp,1e-30),
            'point_err2': err0+err1, 'point_ref2': (x0.square().sum()+x1.square().sum()).item(),
            'point_nmse': (err0+err1)/max((x0.square().sum()+x1.square().sum()).item(),1e-30),
            'scale_switch_fraction': switch.float().mean().item(),
            'switch_block_response_fraction': block_err[switch].sum().item()/max(response_err,1e-30),
            'saturation_fraction': saturated.float().mean().item(),
            'error_cross': (e0*e1).sum().item(), 'n': x0.numel()}


@torch.inference_mode()
def analyze(args):
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []; transitions = []
    state = torch.load(args.checkpoint/'model.pt', map_location='cpu', weights_only=False)
    for pid in args.prompts.split(','):
        payloads = [torch.load(args.output/f'{pid}_s{s}.pt', map_location='cpu', weights_only=False) for s in args.steps]
        for name in LAYERS:
            weight = state[name+'.weight'].cuda().float()
            # model.pt contains dequantized fixed residual weights in smoothed coordinates.
            xs = [p['layers'][name]['x'].cuda().float() for p in payloads]
            maxima = [p['layers'][name]['full_absmax'] for p in payloads]
            pairs = [('denoise', args.steps[s], xs[s], xs[s+1], maxima[s], maxima[s+1])
                     for s in range(len(xs)-1) if args.steps[s+1]==args.steps[s]+1]
            pairs += [('frame', args.steps[s], xs[s].view(-1,16,xs[s].shape[-1])[:-1].reshape(-1,xs[s].shape[-1]),
                       xs[s].view(-1,16,xs[s].shape[-1])[1:].reshape(-1,xs[s].shape[-1]), maxima[s], maxima[s]) for s in range(len(xs))]
            for axis, step, a, b, maxa, maxb in pairs:
                x0 = a.reshape(a.shape[0], -1, 16)
                delta = (b-a).reshape_as(x0)
                rel = (delta.square().mean()/x0.square().mean().clamp_min(1e-30)).sqrt().item()
                g = max(maxa,maxb) / (6*448)
                s0 = scale_for(x0,g); y0 = qdq(x0,s0)
                for eps in [0.001, 0.01, 0.05, 0.1, -1.0]:
                    alpha = 1.0 if eps < 0 else eps/max(rel,1e-30)
                    x1 = x0 + alpha*delta
                    s1 = scale_for(x1,g)
                    candidates = {'dynamic_block': (y0,qdq(x1,s1),s0,s1),
                                  'frozen_block': (y0,qdq(x1,s0),s0,s0)}
                    # Hold previous scale only when its local reconstruction cost is within 5% of dynamic.
                    yf, yd = candidates['frozen_block'][1], candidates['dynamic_block'][1]
                    keep = (yf-x1).square().sum(-1,keepdim=True) <= 1.05*(yd-x1).square().sum(-1,keepdim=True)
                    sh = torch.where(keep,s0,s1)
                    candidates['hysteresis_mse5'] = (y0, torch.where(keep,yf,yd),s0,sh)
                    c0=scale_for(x0,g,True); c1=scale_for(x1,g,True)
                    candidates['continuous_scale_control']=(qdq(x0,c0),qdq(x1,c1),c0,c1)
                    if eps < 0:
                        sg0=scale_for(x0,maxa/(6*448)); sg1=scale_for(x1,maxb/(6*448))
                        candidates['dynamic_global_endpoint']=(qdq(x0,sg0),qdq(x1,sg1),sg0,sg1)
                    for mode,(q0,q1,t0,t1) in candidates.items():
                        row=stats(x0,x1,q0,q1,t0,t1,prompt=pid,layer=name,axis=axis,step=step,
                                  eps=eps,actual_relative_delta=rel,alpha=alpha,mode=mode)
                        # Project numerical error through the same fixed W4 residual; LR branch cancels.
                        if eps in [0.01,0.1,-1.0]:
                            ed=(q1-q0-(x1-x0)).reshape(a.shape)
                            out_err=F.linear(ed,weight)
                            out_ref=F.linear((x1-x0).reshape(a.shape),weight)
                            row['linear_response_err2']=out_err.square().sum().item()
                            row['linear_response_ref2']=out_ref.square().sum().item()
                            row['linear_response_nmse']=row['linear_response_err2']/max(row['linear_response_ref2'],1e-30)
                        rows.append(row)
                    # Exact additive decomposition at fixed global scale: discrete code motion at old
                    # grid + grid replacement. These are NOT independent variance components.
                    elem=qdq(x1,s0)-y0-(x1-x0)
                    grid=qdq(x1,s1)-qdq(x1,s0)
                    transitions.append({'prompt':pid,'layer':name,'axis':axis,'step':step,'eps':eps,
                        'element_err2':elem.square().sum().item(),'grid_err2':grid.square().sum().item(),
                        'cross2':2*(elem*grid).sum().item(),'total_err2':(elem+grid).square().sum().item()})
            print(f'ANALYZED {pid} {name}',flush=True)
            prefix='' if args.rounding=='rne' else 'dc_'
            write_csv(args.output/f'{prefix}local_{args.prompts.replace(",","_")}.csv',rows)
            write_csv(args.output/f'{prefix}decomposition_{args.prompts.replace(",","_")}.csv',transitions)


def main():
    global ROUNDING_MODE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['capture','analyze'])
    ap.add_argument('--prompts',default='0000,0005')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model-path',type=Path,default=MODEL)
    ap.add_argument('--checkpoint',type=Path,default=CKPT)
    ap.add_argument('--cache-dir',type=Path,default=CACHE)
    ap.add_argument('--steps',default='0,1,2,3')
    ap.add_argument('--standard-wan',action='store_true')
    ap.add_argument('--rounding',choices=['rne','deepcompressor'],default='rne')
    args=ap.parse_args();args.steps=[int(s) for s in args.steps.split(',')];args.output.mkdir(parents=True,exist_ok=True)
    ROUNDING_MODE=args.rounding
    globals()[args.phase](args)


if __name__=='__main__':main()
