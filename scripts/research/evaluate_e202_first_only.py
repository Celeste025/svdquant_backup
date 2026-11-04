#!/usr/bin/env python3
"""Score only the new probability-first video against its existing BF16 reference."""
import argparse,json,math
from itertools import zip_longest
from pathlib import Path
import av,numpy as np,torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E202'
def frames(path):
 with av.open(str(path)) as c:
  for frame in c.decode(video=0):yield frame.to_ndarray(format='rgb24')
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--prompt-id',type=int,required=True);a=p.parse_args();pid=a.prompt_id;case=f'p{pid:03d}'
 ref=json.loads((ROOT/'results/research/E150'/case/'bf16_denoise.json').read_text());refv=json.loads((ROOT/'results/research/E150'/case/'bf16_video.json').read_text());d=json.loads((OUT/'first'/case/'bf16_denoise.json').read_text());v=json.loads((OUT/'first'/case/'bf16_video.json').read_text())
 assert ref['status']=='denoise_complete' and refv['status']=='complete' and d['status']=='denoise_complete' and v['status']=='complete' and d['initial_latent_sha256']==ref['initial_latent_sha256'] and d['settings']==ref['settings'] and d['frames']==ref['frames'] and v['frames']==ref['frames']
 torch.set_num_threads(6);metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval();sse=0;lp=[]
 for count,(x,y) in enumerate(zip_longest(frames(refv['video']),frames(v['video'])),1):
  assert x is not None and y is not None and x.shape==y.shape==(768,1344,3)
  diff=x.astype(np.int32)-y.astype(np.int32);sse+=int(np.square(diff).sum(dtype=np.int64))
  def tensor(im):return torch.from_numpy(im).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255)
  lp.append(float(metric(tensor(x),tensor(y))));metric.reset()
 assert count==ref['frames']
 result={'status':'complete','prompt_id':pid,'frames':count,'reference':refv['video'],'video':v['video'],'seed':ref['seed'],'metric':'all RGB8 frames, global RGB PSNR and LPIPS Alex FP32 normalize=True averaged over frames','psnr_db':10*math.log10(255**2*count*768*1344*3/sse),'lpips_alex':sum(lp)/count,'lpips_per_frame':lp,'dit_seconds':sum(x['dit_seconds'] for x in d['steps']),'step_seconds':[x['dit_seconds'] for x in d['steps']],'video_latent_nmse':d['comparison']['video_latent_nmse']}
 path=OUT/f'{case}_first_metrics.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print('E202_FIRST_EVAL_COMPLETE',case,flush=True)
if __name__=='__main__':main()
