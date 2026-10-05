#!/usr/bin/env python3
"""E082 clapping native center128 comparisons."""
import argparse,json,time,sys,itertools
from pathlib import Path
import av,numpy as np,torch
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import evaluate_h3_baseline_distances as old
ROOT=HERE.parents[1];OUT=ROOT/'results/research/E082';DATA=Path('/data1/models/svdquant-wjq/research/20261005/E082')

def triplets():
 import read_h3_sage3_videos as prior
 base={r['case']:r for r in prior.triplets() if r['case'].startswith('vbench0161_')}
 rows=[]
 for replica in (0,1):
  gen=json.loads((OUT/f'denoise_svd_sage3_center128_r{replica}.json').read_text())
  dec=json.loads((OUT/f'decode_svd_sage3_center128_r{replica}.json').read_text())
  launch=json.loads((OUT/f'launcher_r{replica}.json').read_text())
  assert gen['status']==dec['status']==launch['status']=='complete'
  assert gen['complete_dit_calls']==20 and gen['actual_fp4_attention_calls']==1000
  assert len(gen['cases'])==len(dec['cases'])==1
  g=gen['cases'][0];v=dec['cases'][0];row=base[g['case_id']]
  prep=json.loads((ROOT/f'results/research/E079/denoise_svd_sage3_r{replica}.json').read_text())
  original=next(r for r in prep['cases'] if r['case_id']==row['case'])
  assert g['initial_noise_sha256']==original['initial_noise_sha256'] and g['embedding_sha256']==original['embedding_sha256']
  assert v['settings']==row['settings'] and v['seed']==row['seed']
  assert old.sha(v['video_path'])==v['video_sha256']
  row.update(center=v['video_path']);row['sha256']['center']=v['video_sha256'];rows.append(row)
 return rows

def render(rows):
 dest=DATA/'review';dest.mkdir(exist_ok=False);receipts=[]
 for row in rows:
  frames={}
  for a in ('bf16','sage','center'):
   frames[a]=[Image.fromarray(f) for f in old.frames(row[a])];assert len(frames[a])==124
  indices=list(range(0,124,4))+[123];assert len(indices)==32
  pages=[]
  for page in range(4):
   seq=indices[8*page:8*page+8];canvas=Image.new('RGB',(1536,1424),'white');draw=ImageDraw.Draw(canvas)
   draw.text((5,4),row['case']+' | BF16 / SVDQuant+Sage3 / +V-center128 | sample frames',fill='black')
   for j,i in enumerate(seq):
    for n,a in enumerate(('bf16','sage','center')):
     x=(j%4)*384;y=30+(j//4*3+n)*232
     draw.text((x+3,y),f'{a} frame {i}',fill='black');canvas.paste(frames[a][i].resize((384,216)),(x,y+16))
   p=dest/f'{row["case"]}_page{page}.png';canvas.save(p);pages.append(str(p))
  target=dest/f'{row["case"]}_triplet.mp4'
  with av.open(str(target),'w') as container:
   stream=container.add_stream('libx264',rate=24);stream.width=1536;stream.height=312;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast','threads':'4'}
   for i in range(124):
    frame=Image.new('RGB',(1536,312),'white');draw=ImageDraw.Draw(frame)
    for j,(a,label) in enumerate((('bf16','BF16'),('sage','SVDQuant + SageAttention3'),('center','+ V-center128'))):
     draw.text((j*512+5,5),label,fill='black');frame.paste(frames[a][i].resize((512,288)),(j*512,24))
    for packet in stream.encode(av.VideoFrame.from_image(frame)):container.mux(packet)
   for packet in stream.encode():container.mux(packet)
  receipts.append(dict(case=row['case'],pages=pages,frames=indices,comparison_video=str(target),comparison_sha256=old.sha(target)))
  print('rendered',row['case'],flush=True)
 (OUT/'visual_manifest.json').write_text(json.dumps(dict(status='complete',rows=receipts,scope='32 sampled frames/clip; labeled contact sheets; full 124-frame silent triplet at 24fps, scaled512x288 panels'),indent=2))
 html='<meta charset="utf-8"><title>H3 native V-center128 comparison</title><h1>BF16 / SVDQuant + SageAttention3 / + V-center128</h1><p>Same prompt, noise, settings. Silent comparison; original clips retain audio.</p>'
 for r in receipts:html+=f'<h2>{r["case"]}</h2><video controls preload="metadata" width="1200" src="{Path(r["comparison_video"]).name}"></video>'
 (dest/'index.html').write_text(html)

@torch.inference_mode()
def metrics(rows):
 from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
 assert not (OUT/'metrics.json').exists()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(4);metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval()
 allrows=[]
 for row in rows:
  for ref in ('bf16','sage'):
   pair=dict(case=row['case']+'__center_vs_'+ref,label=row['label'],seed=row['seed'],cohort='E082',candidate=row['center'],reference=row[ref],candidate_sha256=row['sha256']['center'],reference_sha256=row['sha256'][ref])
   r=old.evaluate(pair,metric);r['reference_arm']=ref;allrows.append(r)
   (OUT/(pair['case']+'.json')).write_text(json.dumps(r,indent=2));print(pair['case'],r['lpips_alex'],flush=True)
 summaries={ref:{k:float(np.mean([r[k] for r in allrows if r['reference_arm']==ref])) for k in ('lpips_alex','l1_mae','l2_rmse')} for ref in ('bf16','sage')}
 (OUT/'metrics.json').write_text(json.dumps(dict(status='complete',definition='All124frames original1024x576 RGB[0,1], LPIPS-Alex FP32 and MAE/RMSE; pair distance not quality;2clips equal-weight',summary=summaries,rows=allrows),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['render','metrics'],required=True);a=p.parse_args();start=time.time();rows=triplets()
 (OUT/'triplets.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 (render if a.phase=='render' else metrics)(rows)
 print('complete',a.phase,time.time()-start,flush=True)
