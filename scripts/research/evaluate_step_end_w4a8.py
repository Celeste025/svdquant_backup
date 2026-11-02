#!/usr/bin/env python3
"""Compare W4A8 endpoint video against matched BF16 reference."""
import json, math
from itertools import zip_longest
from pathlib import Path
import av
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/research/step_end_w4a8'; case='p016'; arm='ends_nvfp4_int8_g128'
base=json.loads((ROOT/'results/research/E136/p016_video_metrics.json').read_text()); d=json.loads((OUT/case/f'{arm}_denoise.json').read_text()); v=json.loads((OUT/case/f'{arm}_video.json').read_text()); assert d['initial_latent_sha256']==base['initial_latent_sha256']
def frames(path):
 with av.open(str(path)) as c:
  for f in c.decode(video=0): yield f.to_ndarray(format='rgb24')
metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval(); sse=energy=0; lp=[]
with torch.inference_mode():
 for n,(ref,im) in enumerate(zip_longest(frames(base['sources']['bf16']),frames(v['video'])),1):
  assert ref is not None and im is not None and ref.shape==im.shape==(576,1024,3)
  delta=im.astype(np.int32)-ref.astype(np.int32);sse+=int(np.square(delta).sum(dtype=np.int64));energy+=int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
  def t(x):return torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255)
  lp.append(float(metric(t(im),t(ref))));metric.reset()
assert n==124
r={'status':'complete','case':case,'arm':arm,'frames':n,'initial_latent_sha256':d['initial_latent_sha256'],'reference':base['sources']['bf16'],'video':v['video'],'metrics':{'rgb_nmse':sse/energy,'lpips_alex_mean':float(np.mean(lp)),'psnr_db_global':10*math.log10(255**2*124*576*1024*3/sse)},'method':'RGB8 all 124 frames; LPIPS AlexNet FP32 normalize=True mean; global RGB PSNR'}
(OUT/'p016_video_metrics.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
