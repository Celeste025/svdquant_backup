#!/usr/bin/env python3
"""Full-frame PSNR and LPIPS for first-step route capture/reuse."""
import json,math
from itertools import zip_longest
from pathlib import Path
import av,numpy as np,torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E201'
def frames(path):
 with av.open(str(path)) as c:
  for f in c.decode(video=0):yield f.to_ndarray(format='rgb24')
@torch.inference_mode()
def main():
 torch.set_num_threads(6);ref=json.loads((ROOT/'results/research/E198/bf16.json').read_text());assert ref['status']=='complete'
 metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval()
 report={'status':'running','reference':ref['video'],'method':'RGB8 all 124 frames, global RGB PSNR, LPIPS AlexNet FP32 normalize=True averaged over frames','results':[]}
 for rule in ('reuse_qk','reuse_prob'):
  for skip in (0,75):
   d=json.loads((OUT/f'{rule}_skip{skip}.json').read_text());assert d['status']=='complete' and d['settings']==ref['settings'] and d['initial_latent_sha256']==ref['initial_latent_sha256']
   sse=0;lp=[]
   for n,(a,b) in enumerate(zip_longest(frames(ref['video']),frames(d['video'])),1):
    assert a is not None and b is not None and a.shape==b.shape==(480,832,3)
    delta=a.astype(np.int32)-b.astype(np.int32);sse+=int(np.square(delta).sum(dtype=np.int64))
    def t(x):return torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255)
    lp.append(float(metric(t(a),t(b))));metric.reset()
   assert n==124
   row={'rule':rule,'skip':skip/100,'video':d['video'],'psnr_db':10*math.log10(255**2*n*480*832*3/sse),'lpips_alex_mean':sum(lp)/len(lp),'lpips_per_frame':lp}
   report['results'].append(row);(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in row.items() if k!='lpips_per_frame'},flush=True)
 report['status']='complete';(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
