#!/usr/bin/env python3
"""Plot all forward overlaps for per-step H3 softmax-mean maps."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/research/tile_probability_overlap_20261103'
maps=np.load(OUT/'step_maps.npz')['maps']; steps,heads,n,_=maps.shape
assert steps==20 and heads==56 and n==148
flat=maps.reshape(steps,heads,-1);total=n*n
rates=(.1,.2,.3);matrices=[]
for rate in rates:
 k=math.floor(rate*total+.5);order=np.argsort(-flat,axis=-1,kind='stable')[:,:,:k]
 selected=np.zeros((steps,heads,total),dtype=bool)
 np.put_along_axis(selected,order,True,axis=-1)
 overlap=np.empty((steps,steps,heads),dtype=np.float32)
 for i in range(steps):
  for j in range(steps):overlap[i,j]=(selected[i]&selected[j]).sum(axis=-1)/k
 assert np.allclose(overlap,overlap.transpose(1,0,2)) and np.allclose(overlap[np.arange(steps),np.arange(steps)],1)
 matrices.append(overlap)
fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True,layout='constrained')
cmap=plt.get_cmap('turbo');norm=Normalize(0,18)
for ax,rate,mat in zip(axs,rates,matrices):
 for start in range(19):ax.plot(np.arange(start,steps),mat[start,start:].mean(axis=-1)*100,color=cmap(norm(start)),linewidth=1.6,alpha=.9)
 ax.set(title=f'Top {int(rate*100)}% tiles',xlabel='Later step (zero-based)',xlim=(0,19),ylim=(0,102),xticks=[0,5,10,15,19]);ax.grid(alpha=.2)
axs[0].set_ylabel('Overlap with start step (%)')
fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs,label='Start step',ticks=[0,3,6,9,12,15,18],shrink=.83)
fig.suptitle('Forward stability of softmax-mean tile selection | H3 layer 25',fontsize=14)
fig.text(.04,-.025,'56-head mean · BF16 dense trajectory · Q128 × K128 · prompt 16, seed 20026',fontsize=10)
fig.savefig(OUT/'forward_step_overlap.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'forward_step_overlap.pdf',bbox_inches='tight')
summary={'status':'complete','definition':'global per-head top-k tile intersection / k; averages over 56 heads','layer_zero_based':25,'prompt_id':16,'seed':20026,'shape_steps_heads_query_tiles_key_tiles':list(maps.shape),'rates':{str(rate):{'tiles_per_head':math.floor(rate*total+.5),'mean_by_start_and_later_step':m.mean(axis=-1).tolist(),'overlap_by_start_later_head':m.tolist()} for rate,m in zip(rates,matrices)}}
(OUT/'forward_step_overlap.json').write_text(json.dumps(summary,indent=2)+'\n')
print('created figures and all-pair data')
