#!/usr/bin/env python3
"""Paired full-video NMSE, LPIPS, and comparison montage for E147."""
import json
from itertools import zip_longest
from pathlib import Path
import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/research/E147/p025'
DATA=Path('/data1/models/svdquant-wjq/research/20261101/E147_paper_pair/p025')
records=[(ROOT/'results/research/E124/p025','bf16'),(ROOT/'results/research/E146/p025','s80_int8_20'),(OUT,'s743_fp4193_int864')]

@torch.inference_mode()
def main():
    den=[json.loads((d/f'{a}_denoise.json').read_text()) for d,a in records]
    vids=[json.loads((d/f'{a}_video.json').read_text()) for d,a in records]
    assert all(x['status']=='denoise_complete' and len(x['steps'])==20 for x in den)
    assert all(x['initial_latent_sha256']==den[0]['initial_latent_sha256'] and x['settings']==den[0]['settings'] for x in den)
    metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval()
    containers=[av.open(x['video']) for x in vids]
    montage=DATA/'p025_bf16_80int8_paper_mixed.mp4'
    assert not montage.exists()
    stats=[{'sse':0,'energy':0,'lpips':[]} for _ in range(2)]
    selected=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    try:
        with av.open(str(montage),'w',options={'movflags':'+faststart'}) as output:
            stream=output.add_stream('libx264',rate=24);stream.width=1536;stream.height=320;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
            for idx, group in enumerate(zip_longest(*(c.decode(video=0) for c in containers))):
                assert idx<124 and all(f is not None for f in group)
                images=[f.to_ndarray(format='rgb24') for f in group]
                assert all(x.shape==(576,1024,3) for x in images)
                ref=images[0];ty=torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).cuda().float()/255
                canvas=Image.new('RGB',(1536,320),'black');draw=ImageDraw.Draw(canvas)
                third_label = ('75 skip / 15 FP4 / 10 INT8' if records[2][1] == 's75_fp415_int810'
                               else '74.3 skip / 19.3 FP4 / 6.4 INT8')
                for col,(im,label) in enumerate(zip(images,['BF16','80 skip / 20 INT8',third_label])):
                    canvas.paste(Image.fromarray(im).resize((512,288)),(col*512,32));draw.text((col*512+6,6),label,font=font,fill='white')
                    if col:
                        delta=im.astype(np.int32)-ref.astype(np.int32); row=stats[col-1]
                        row['sse']+=int(np.square(delta).sum(dtype=np.int64));row['energy']+=int(np.square(ref.astype(np.int32)).sum(dtype=np.int64))
                        tx=torch.from_numpy(im).permute(2,0,1).unsqueeze(0).cuda().float()/255
                        row['lpips'].append(float(metric(tx,ty)));metric.reset()
                if idx in (0,31,62,93,123):selected.append(canvas.copy())
                for pkt in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):output.mux(pkt)
            assert idx==123
            for pkt in stream.encode():output.mux(pkt)
    finally:
        for c in containers:c.close()
    sheet=Image.new('RGB',(1536,320*len(selected)))
    for i,im in enumerate(selected):sheet.paste(im,(0,320*i))
    sheet.save(OUT/'comparison_frames.jpg',quality=94)
    result={'prompt_id':25,'seed':2141,'initial_latent_sha256':den[0]['initial_latent_sha256'],'sources':{a:v['video'] for (_,a),v in zip(records,vids)},'dit_seconds':{a:sum(s['dit_seconds'] for s in d['steps']) for (_,a),d in zip(records,den)},'metrics':{records[i+1][1]:{'rgb_nmse':r['sse']/r['energy'],'lpips_alex_mean':float(np.mean(r['lpips']))} for i,r in enumerate(stats)},'comparison_video':str(montage)}
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
