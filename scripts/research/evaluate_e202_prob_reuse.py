#!/usr/bin/env python3
"""All-frame RGB PSNR and LPIPS Alex for four E202 cases."""
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
 ref=json.loads((ROOT/'results/research/E150'/case/'bf16_denoise.json').read_text());refv=json.loads((ROOT/'results/research/E150'/case/'bf16_video.json').read_text());assert ref['status']=='denoise_complete' and refv['status']=='complete'
 modes=('first','periodic','baseline');ds=[json.loads((OUT/m/case/'bf16_denoise.json').read_text()) for m in modes];vs=[json.loads((OUT/m/case/'bf16_video.json').read_text()) for m in modes]
 for d,v in zip(ds,vs):assert d['status']=='denoise_complete' and v['status']=='complete' and d['initial_latent_sha256']==ref['initial_latent_sha256'] and d['settings']==ref['settings'] and d['frames']==ref['frames'] and v['frames']==ref['frames']
 torch.set_num_threads(6);metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval();rows=[{'sse':0,'lpips_per_frame':[]} for _ in modes]
 for count,group in enumerate(zip_longest(*(frames(x) for x in (refv['video'],*(v['video'] for v in vs)))),1):
  assert all(x is not None and x.shape==(768,1344,3) for x in group)
  refimg=group[0];ref_t=torch.from_numpy(refimg).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255)
  for image,row in zip(group[1:],rows):
   diff=image.astype(np.int32)-refimg.astype(np.int32);row['sse']+=int(np.square(diff).sum(dtype=np.int64))
   sample=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to('cuda',torch.float32).div_(255);row['lpips_per_frame'].append(float(metric(sample,ref_t)));metric.reset()
 assert count==ref['frames']
 result={'status':'complete','prompt_id':pid,'frames':count,'reference':refv['video'],'seed':ref['seed'],'metric':'all RGB8 frames, global RGB PSNR and LPIPS Alex FP32 normalize=True averaged over frames','policies':{}}
 for m,d,v,row in zip(modes,ds,vs,rows):
  result['policies'][m]={'video':v['video'],'psnr_db':10*math.log10(255**2*count*768*1344*3/row['sse']) if row['sse'] else None,'lpips_alex':sum(row['lpips_per_frame'])/count,'lpips_per_frame':row['lpips_per_frame'],'dit_seconds':sum(x['dit_seconds'] for x in d['steps']),'step_seconds':[x['dit_seconds'] for x in d['steps']],'video_latent_nmse':d['comparison']['video_latent_nmse']}
 path=OUT/f'{case}_metrics.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print('E202_EVAL_COMPLETE',case,flush=True)
if __name__=='__main__':main()
