#!/usr/bin/env python3
"""Plot relative change in pooled-head video output NMSE across NVFP4 budgets."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/E214';pids=(25,7,48,44);names=('Mahjong cats','Mars astronaut','Kayak','Swimmer');x=np.full((4,20,50,3),np.nan);lookup={p:i for i,p in enumerate(pids)}
for r in csv.DictReader((out/'step_layer_summary.csv').open()):
 p=lookup[int(r['prompt_id'])];s=int(r['step']);l=int(r['layer']);assert np.isnan(x[p,s,l]).all();x[p,s,l]=[float(r[f'overall_nmse_{i}']) for i in (10,40,90)]
assert np.isfinite(x).all() and (x>0).all()
pairs=((0,1),(1,2),(0,2));labels=('10% → 40% NVFP4','40% → 90% NVFP4','10% → 90% NVFP4');change=np.stack([(x[...,j]/x[...,i]-1)*100 for i,j in pairs],-1)
assert change.min()>=-100 and change.max()<=100
fig,axs=plt.subplots(4,3,figsize=(15,13),sharex=True,sharey=True,layout='constrained');norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100)
for p in range(4):
 for j in range(3):
  ax=axs[p,j];im=ax.imshow(change[p,:,:,j].T,origin='lower',aspect='auto',interpolation='nearest',extent=(-.5,19.5,-.5,49.5),cmap='RdBu_r',norm=norm)
  if p==0:ax.set_title(labels[j])
  if j==0:ax.set_ylabel(f'p{pids[p]:03d} · {names[p]}\nLayer (zero-based)')
  if p==3:ax.set_xlabel('Step (zero-based)')
fig.colorbar(im,ax=axs,label='Relative change in pooled-head NMSE (%)',shrink=.8,ticks=[-100,-75,-50,-25,0,25,50,75,100]);fig.suptitle('Whole-layer error change as NVFP4 fraction increases | Q64 × K64, fixed 10% INT8',fontsize=15)
fig.savefig(out/'layer_nmse_relative_change.png',dpi=170);fig.savefig(out/'layer_nmse_relative_change.pdf');plt.close(fig)
with (out/'layer_nmse_relative_change.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['prompt_id','step','layer','nmse_10','nmse_40','nmse_90','relative_percent_10_to_40','relative_percent_40_to_90','relative_percent_10_to_90'])
 for p,pid in enumerate(pids):
  for s in range(20):
   for l in range(50):w.writerow([pid,s,l,*x[p,s,l].tolist(),*change[p,s,l].tolist()])
summary={'definition':'100*(NMSE_higher/NMSE_lower - 1), pooled 56-head video-query error energy / pooled BF16 reference energy; positive = error increases','min_reference_nmse':float(x.min()),'pairs':{label:{'median_percent':float(np.median(change[...,j])),'min_percent':float(change[...,j].min()),'max_percent':float(change[...,j].max()),'positive_cells':int((change[...,j]>0).sum()),'mean_by_layer':change[...,j].mean((0,1)).tolist()} for j,label in enumerate(labels)}}
(out/'layer_nmse_relative_change_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='pairs'},indent=2))
