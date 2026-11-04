#!/usr/bin/env python3
"""Visualize true softmax-mean top-tile routing and precision incidence."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[2];src=ROOT/'results/research/tile_probability_overlap_q64_20261103';out=ROOT/'results/research/E216';heads=(10,20,30);rates=(.1,.2,.3);steps=range(20);rows=[];masks={}
for step in steps:
 with np.load(src/f'top_step_{step:02d}.npz') as data:
  top=data['true'];nb=int(data['blocks']);total=nb*nb
 for rate in rates:
  n=int(rate*total+.5);mask=np.zeros((56,total),dtype=bool);np.put_along_axis(mask,top[:,:n],True,axis=1);mask=mask.reshape(56,nb,nb)
  for axis,dim in (('Q',-1),('KV',-2)):
   counts=mask.sum(axis=dim);both=(counts>0)&(counts<nb);only_int8=counts==nb;only_fp4=counts==0
   rows.append({'step':step,'int8_fraction':rate,'operand':axis,'total_blocks':counts.size,'both':int(both.sum()),'int8_only':int(only_int8.sum()),'nvfp4_only':int(only_fp4.sum()),'both_fraction':float(both.mean()),'int8_only_fraction':float(only_int8.mean()),'nvfp4_only_fraction':float(only_fp4.mean())})
  if step==0:masks[rate]=mask[np.array(heads)]
fig,axs=plt.subplots(3,3,figsize=(13,12),sharex=True,sharey=True,layout='constrained');cmap=ListedColormap(['#dce8f1','#dc3f33'])
for i,h in enumerate(heads):
 for j,rate in enumerate(rates):
  m=masks[rate][i];qboth=((m.sum(1)>0)&(m.sum(1)<nb)).mean()*100;kboth=((m.sum(0)>0)&(m.sum(0)<nb)).mean()*100
  ax=axs[i,j];ax.imshow(m,origin='lower',aspect='equal',interpolation='nearest',cmap=cmap,vmin=0,vmax=1)
  if i==0:ax.set_title(f'True top {int(100*rate)}% → INT8\nrest → NVFP4')
  if j==0:ax.set_ylabel(f'Head {h}\nQuery block')
  if i==2:ax.set_xlabel('Key block')
  ax.text(.02,.98,f'Q both: {qboth:.1f}%\nK/V both: {kboth:.1f}%',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
fig.suptitle('True softmax-mean tile routing | step 0, layer 25, Q64 × K64',fontsize=15)
fig.legend(handles=[Patch(color='#dc3f33',label='INT8'),Patch(color='#dce8f1',label='NVFP4')],loc='outside lower center',ncol=2,frameon=False)
fig.savefig(out/'true_top_route_examples.png',dpi=170);fig.savefig(out/'true_top_route_examples.pdf');plt.close(fig)
with (out/'true_route_precision_incidence.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'status':'complete','source':'E214 context BF16 dense p16 seed 20026, layer25, steps 0-19, true FP32 softmax-mean maps from prior Q64 capture; video tiles only, no anchors, no skip','definitions':'for each video query row or key column: both means incident to at least one INT8 and one NVFP4 tile; only means video interactions exclusively that type. Prefix INT8 is excluded: every video query also needs INT8 when prefix is INT8.','rates':{}}
for rate in rates:
 r={}
 for typ in ('Q','KV'):
  data=[x for x in rows if x['operand']==typ and x['int8_fraction']==rate];r[typ]={k:{'mean':float(np.mean([z[k] for z in data])),'min':float(np.min([z[k] for z in data])),'max':float(np.max([z[k] for z in data]))} for k in ('both_fraction','int8_only_fraction','nvfp4_only_fraction')}
 report['rates'][str(rate)]=r
(out/'true_route_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
