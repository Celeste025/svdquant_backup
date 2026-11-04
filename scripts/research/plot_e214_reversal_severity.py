#!/usr/bin/env python3
"""Highlight large paired NVFP4 NMSE reversals by prompt, step and layer."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/E214';pids=(25,7,48,44);names=('Mahjong cats','Mars astronaut','Kayak','Swimmer');delta=np.empty((4,20,50,56));source=np.empty((4,20,50,56),np.uint8)
for pi,pid in enumerate(pids):
 seen=set()
 for line in (out/f'p{pid:03d}'/'cells.jsonl').open():
  r=json.loads(line);s,l=r['step'],r['layer'];assert (s,l) not in seen;seen.add((s,l))
  a=np.array([arm['nmse_by_head'] for arm in r['arms']]);diff=np.stack((a[1]-a[0],a[2]-a[1],a[2]-a[0]))*100
  delta[pi,s,l]=np.maximum(diff.max(0),0);source[pi,s,l]=diff.argmax(0)
 assert len(seen)==1000
threshold=.1;severe=delta>threshold;maxdelta=delta.max(-1);fraction=severe.mean(-1)*100
assert np.isfinite(delta).all()
fig,axs=plt.subplots(2,4,figsize=(19,8),sharex=True,sharey=True,layout='constrained')
cmap=plt.get_cmap('magma').copy();cmap.set_bad('#e9e9e9')
for p in range(4):
 ax=axs[0,p];im=ax.imshow(np.ma.masked_less_equal(maxdelta[p].T,threshold),origin='lower',aspect='auto',interpolation='nearest',extent=(-.5,19.5,-.5,49.5),cmap=cmap,norm=LogNorm(vmin=threshold,vmax=max(10.,float(maxdelta.max()))));ax.set_title(f'p{pids[p]:03d} · {names[p]}');ax.set_xlabel('Step (zero-based)')
 ax=axs[1,p];im2=ax.imshow(fraction[p].T,origin='lower',aspect='auto',interpolation='nearest',extent=(-.5,19.5,-.5,49.5),cmap='viridis',vmin=0,vmax=25);ax.set_xlabel('Step (zero-based)')
axs[0,0].set_ylabel('Layer: maximum increase');axs[1,0].set_ylabel('Layer: fraction of heads')
fig.colorbar(im,ax=axs[0,:],label='Max increase (NMSE pp, log scale)',shrink=.83);fig.colorbar(im2,ax=axs[1,:],label='Heads above 0.1 NMSE pp (%)',shrink=.83)
fig.suptitle('Severe NVFP4 error reversals | 10% INT8; compare 10%, 40%, 90% NVFP4',fontsize=15)
fig.savefig(out/'reversal_severity.png',dpi=160);fig.savefig(out/'reversal_severity.pdf');plt.close(fig)
with (out/'reversal_severity_by_cell.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['prompt_id','step','layer','max_increase_nmse_pp','max_head','max_pair','heads_above_0.1pp','heads_fraction_above_0.1pp'])
 for p,pid in enumerate(pids):
  for s in range(20):
   for l in range(50):
    h=int(delta[p,s,l].argmax());w.writerow([pid,s,l,float(maxdelta[p,s,l]),h,['10_to_40','40_to_90','10_to_90'][source[p,s,l,h]],int(severe[p,s,l].sum()),float(fraction[p,s,l]/100)])
summary={'threshold_pp':threshold,'severity_definition':'max(0, NMSE40-NMSE10, NMSE90-NMSE40, NMSE90-NMSE10) in percentage points, per head, same BF16 reference','max_pp_by_prompt':dict(zip(map(str,pids),map(float,maxdelta.max((1,2))))),'severe_count_by_layer':severe.sum((0,1,3)).tolist(),'severe_count_by_step':severe.sum((0,2,3)).tolist(),'severe_count_by_prompt':dict(zip(map(str,pids),map(int,severe.sum((1,2,3))))),'observations_above_threshold':int(severe.sum()),'max_by_layer':maxdelta.max((0,1)).tolist(),'total_head_observations':int(severe.size)}
(out/'reversal_severity_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
