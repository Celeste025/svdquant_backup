#!/usr/bin/env python3
"""Full-frame BF16-paired LPIPS Alex and PSNR, plus side-by-side video."""
import argparse,json,math
from itertools import zip_longest
from pathlib import Path
import av,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research/E162'
VIDEO=ROOT/'results/local/E162/comparisons'
@torch.inference_mode()
def main(pid):
 torch.set_num_threads(6);case=f'p{pid:03d}'
 refs=[ROOT/'results/research/E150'/case/'bf16_video.json', ROOT/'results/research/E159/aligned_v2/proxy'/case/'bf16_video.json', ROOT/'results/research/E162/proxy_new'/case/'bf16_video.json']
 v=[json.loads(p.read_text()) for p in refs]
 den=[json.loads((ROOT/'results/research/E150'/case/'bf16_denoise.json').read_text()), json.loads((ROOT/'results/research/E159/aligned_v2/proxy'/case/'bf16_denoise.json').read_text()), json.loads((ROOT/'results/research/E162/proxy_new'/case/'bf16_denoise.json').read_text())]
 assert all(x['status']=='denoise_complete' and x['initial_latent_sha256']==den[0]['initial_latent_sha256'] and x['settings']==den[0]['settings'] and x['frames']==den[0]['frames'] and x['seed']==den[0]['seed'] for x in den)
 assert all(x['status']=='complete' and x['frames']==den[0]['frames'] and x['width']==1344 and x['height']==768 for x in v)
 metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval()
 containers=[av.open(x['video']) for x in v]; VIDEO.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 target=VIDEO/f'{case}_bf16_proxy_proxy_new.mp4';assert not target.exists()
 sums=[{'sse':0,'lpips':[]} for _ in range(2)]; font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);samples=[]
 try:
  with av.open(str(target),'w',options={'movflags':'+faststart'}) as out:
   stream=out.add_stream('libx264',rate=24);stream.width=1536;stream.height=326;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
   for idx,group in enumerate(zip_longest(*(c.decode(video=0) for c in containers))):
    assert all(f is not None for f in group) and idx<den[0]['frames']
    imgs=[f.to_ndarray(format='rgb24') for f in group];assert all(im.shape==(768,1344,3) for im in imgs)
    ref=imgs[0];tr=torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
    canvas=Image.new('RGB',(1536,326),'black');draw=ImageDraw.Draw(canvas)
    for col,(im,label) in enumerate(zip(imgs,['BF16 reference','Proxy protected','Proxy new all mixed'])):
     canvas.paste(Image.fromarray(im).resize((512,293)),(col*512,33));draw.text((col*512+8,7),label,font=font,fill='white')
     if col:
      delta=im.astype(np.int32)-ref.astype(np.int32);sums[col-1]['sse']+=int(np.square(delta).sum(dtype=np.int64))
      tx=torch.from_numpy(im).permute(2,0,1).unsqueeze(0).cuda().float().div_(255);sums[col-1]['lpips'].append(float(metric(tx,tr)));metric.reset()
    if idx in (0,den[0]['frames']//4,den[0]['frames']//2,3*den[0]['frames']//4,den[0]['frames']-1):samples.append(canvas.copy())
    for pkt in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):out.mux(pkt)
   assert idx+1==den[0]['frames']
   for pkt in stream.encode():out.mux(pkt)
 finally:
  for c in containers:c.close()
 sheet=Image.new('RGB',(1536,326*len(samples)))
 for i,im in enumerate(samples):sheet.paste(im,(0,326*i))
 sheet.save(OUT/f'{case}_frames.jpg',quality=94)
 count=den[0]['frames']*768*1344*3
 metrics={}
 for name,row in zip(('proxy','proxy_new'),sums):
  mse=row['sse']/count/(255**2);metrics[name]={'psnr_db':10*math.log10(1/mse) if mse else None,'mse':mse,'lpips_alex':float(np.mean(row['lpips'])),'lpips_per_frame':row['lpips']}
 result={'status':'complete','prompt_id':pid,'frames':den[0]['frames'],'fps':24,'width':1344,'height':768,'sources':{k:x['video'] for k,x in zip(('bf16','proxy','proxy_new'),v)},'comparison_video':str(target),'contact_sheet':str(OUT/f'{case}_frames.jpg'),'method':'All frames, original decoded RGB8 1344x768, paired with BF16. LPIPS TorchMetrics AlexNet pretrained normalize=True FP32, mean across frames; PSNR=10log10(1/global MSE) for RGB normalized [0,1].','metrics':metrics}
 (OUT/f'{case}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='metrics'}|{'metrics':{k:{n:x for n,x in m.items() if n!='lpips_per_frame'} for k,m in metrics.items()}},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);main(p.parse_args().pid)
