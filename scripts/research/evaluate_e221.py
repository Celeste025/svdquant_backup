#!/usr/bin/env python3
import argparse,json,math
from itertools import zip_longest
from pathlib import Path
import numpy as np,torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from evaluate_e202_first_only import frames
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--prompt-id',type=int,required=True);p.add_argument('--policy',required=True);a=p.parse_args();case=f'p{a.prompt_id:03d}';out=ROOT/'results/research/E221'/a.policy/case
ref=json.loads((ROOT/'results/research/E150'/case/'bf16_denoise.json').read_text());rv=json.loads((ROOT/'results/research/E150'/case/'bf16_video.json').read_text());d=json.loads((out/'bf16_denoise.json').read_text());v=json.loads((out/'bf16_video.json').read_text())
assert all(d[k]==ref[k] for k in ('seed','settings','frames','initial_latent_sha256')) and v['frames']==ref['frames']
torch.set_num_threads(6);metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval();sse=0;lp=[]
with torch.inference_mode():
 for count,(x,y) in enumerate(zip_longest(frames(rv['video']),frames(v['video'])),1):
  assert x is not None and y is not None and x.shape==y.shape==(768,1344,3)
  diff=x.astype(np.int32)-y.astype(np.int32);sse+=int(np.square(diff).sum(dtype=np.int64))
  def tensor(im):return torch.from_numpy(im).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255)
  lp.append(float(metric(tensor(x),tensor(y))));metric.reset()
assert count==ref['frames']
result={'status':'complete','prompt_id':a.prompt_id,'policy':a.policy,'frames':count,'seed':ref['seed'],'reference':rv['video'],'video':v['video'],'metric':'all RGB8 frames, global RGB PSNR and LPIPS Alex FP32 normalize=True averaged over frames','psnr_db':10*math.log10(255**2*count*768*1344*3/sse),'lpips_alex':sum(lp)/count,'lpips_per_frame':lp,'dit_seconds':sum(x['dit_seconds'] for x in d['steps']),'step_seconds':[x['dit_seconds'] for x in d['steps']],'video_latent_nmse':d['comparison']['video_latent_nmse']}
path=out/'metrics.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print('E221_EVAL_COMPLETE',a.policy,case,flush=True)
