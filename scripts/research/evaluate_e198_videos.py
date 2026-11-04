#!/usr/bin/env python3
"""Compute full-frame RGB PSNR and LPIPS-Alex for E198 closed-loop videos."""
import json, math, sys, time
from itertools import zip_longest
from pathlib import Path
import av, numpy as np, torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/research/E198'
RULES=('draftmap','qk_abs_mean','prob_mean','prob_rowmax_sum','qk_abs_top20_sum','prob_top20_sum','random')
def frames(path):
    with av.open(str(path)) as c:
        for f in c.decode(video=0): yield f.to_ndarray(format='rgb24')
@torch.inference_mode()
def main():
    torch.set_num_threads(6);ref=json.loads((OUT/'bf16.json').read_text());assert ref['status']=='complete'
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval()
    result={'status':'running','reference':ref['video'],'initial_latent_sha256':ref['initial_latent_sha256'],'method':'RGB8 all 124 frames, global RGB PSNR; LPIPS AlexNet FP32 normalize=True, mean over frames','results':[]}
    for rule in RULES:
        for skip in (75,0):
            d=json.loads((OUT/f'{rule}_skip{skip}.json').read_text());assert d['status']=='complete' and d['initial_latent_sha256']==ref['initial_latent_sha256'] and d['settings']==ref['settings']
            sse=0;lp=[]
            for n,(x,y) in enumerate(zip_longest(frames(ref['video']),frames(d['video'])),1):
                assert x is not None and y is not None and x.shape==y.shape==(480,832,3)
                delta=x.astype(np.int32)-y.astype(np.int32);sse+=int(np.square(delta).sum(dtype=np.int64))
                def tensor(z):return torch.from_numpy(z).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255)
                lp.append(float(metric(tensor(x),tensor(y))));metric.reset()
            assert n==124
            res={'rule':rule,'skip':skip/100,'video':d['video'],'psnr_db':10*math.log10(255**2*n*480*832*3/sse) if sse else None,'lpips_alex_mean':sum(lp)/len(lp),'lpips_per_frame':lp}
            result['results'].append(res);(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='lpips_per_frame'}),flush=True)
    result['status']='complete';result['completed']=time.time();(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
