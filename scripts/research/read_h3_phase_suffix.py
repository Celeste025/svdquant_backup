#!/usr/bin/env python3
"""E075 fixed readout; reuse the existing full-frame distance evaluator."""
import json,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw
import evaluate_h3_baseline_distances as ev
R=Path(__file__).resolve().parents[2];D=R/'results/research/E075/v2'
assert json.loads((D/'launcher.json').read_text())['status']=='complete'
assert not (D/'readout.json').exists()
start=time.monotonic();torch.set_num_threads(4)
run=json.loads((D/'run.json').read_text());dec=json.loads((D/'decode.json').read_text())
paths={'phase0':Path('/data1/models/svdquant-wjq/research/20261002/E017/decode_block_mean/p036_block_mean.mp4')}
for row in dec['cases']:
 rec=row['video'];assert ev.sha(rec['file'])==rec['sha256'];paths[row['arm']]=Path(rec['file'])
report={'status':'running','trajectory':{},'pairs':[],'sources':{k:{'file':str(p),'sha256':ev.sha(p)} for k,p in paths.items()}}
arms={a['arm']:a for a in run['arms']}
for arm in ('phase64','oracle'):
 rows=[]
 for a,b in zip(arms[arm]['steps'],arms['phase0']['steps']):
  x=torch.load(a['artifact']['file'],map_location='cpu',weights_only=True);y=torch.load(b['artifact']['file'],map_location='cpu',weights_only=True)
  row={'step':a['step']}
  for m in ('video','audio'):
   for key in ('noise_pred','latents_after'):
    v=x[m][key].double();ref=y[m][key].double();delta=v-ref
    row[m+'_'+key]={'rmse':float(delta.square().mean().sqrt()),'relative_l2':float(delta.norm()/ref.norm())}
  rows.append(row)
 report['trajectory'][arm]=rows
metric=ev.LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval()
for arm in ('phase64','oracle'):
 p=dict(case=arm,label=arm,cohort='E075_single_patch',seed=59526,candidate=str(paths[arm]),reference=str(paths['phase0']))
 row=ev.evaluate(p,metric);report['pairs'].append(row)
indices=[0,24,48,72,96,123]
for arm,p in paths.items():
 canvas=Image.new('RGB',(3*512,2*310),'white');draw=ImageDraw.Draw(canvas)
 for i,frame in enumerate(ev.frames(p)):
  if i in indices:
   n=indices.index(i);x=n%3*512;y=n//3*310
   canvas.paste(Image.fromarray(frame).resize((512,288)),(x,y));draw.text((x+5,y+290),f'{arm}, frame {i}',fill='black')
 canvas.save(D/(arm+'_contact.png'))
report.update(status='complete',seconds=time.monotonic()-start)
(D/'readout.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({r['case']:{k:r[k] for k in ('lpips_alex','l1_mae','l2_rmse')} for r in report['pairs']},indent=2))
