#!/usr/bin/env python3
"""Plot Q64 same-step DraftMap and forward softmax top-tile overlap."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/tile_probability_overlap_q64_20261103'
true=[];draft=[]
for s in range(20):
 with np.load(out/f'top_step_{s:02d}.npz') as d:
  true.append(d['true']);draft.append(d['draft']);n=int(d['blocks']);prefix=int(d['prefix'])
true=np.stack(true);draft=np.stack(draft);steps,heads,_=true.shape;assert (steps,heads)==(20,56)
total=n*n;rates=(.1,.2,.3);results={}
fig,axs=plt.subplots(1,3,figsize=(15,5.3),sharey=True,layout='constrained');cmap=plt.get_cmap('turbo');norm=Normalize(0,18)
for ax,rate in zip(axs,rates):
 k=math.floor(rate*total+.5);overlap=np.empty((steps,steps,heads),np.float32);dr=np.empty((steps,heads),np.float32)
 for h in range(heads):
  selected=np.zeros((steps,total),bool);np.put_along_axis(selected,true[:,h,:k],True,axis=1)
  for i in range(steps):
   overlap[i,:,h]=(selected & selected[i]).sum(axis=1)/k
   dr[i,h]=selected[i,draft[i,h,:k]].mean()
 assert np.allclose(overlap,overlap.transpose(1,0,2)) and np.allclose(overlap[np.arange(steps),np.arange(steps)],1)
 mean=overlap.mean(-1);base=dr.mean(-1)
 results[str(rate)]={'tiles_per_head':k,'reuse_mean_by_step_pair':mean.tolist(),'reuse_by_step_pair_and_head':overlap.tolist(),'draft_mean_by_step':base.tolist(),'draft_by_step_and_head':dr.tolist()}
 for start in range(19):ax.plot(np.arange(start,steps),mean[start,start:]*100,color=cmap(norm(start)),linewidth=1.35,alpha=.72)
 ax.plot(np.arange(steps),base*100,color='black',linestyle='--',linewidth=2.6,zorder=5)
 ax.set(title=f'Top {int(rate*100)}% tiles',xlabel='Later / current step (zero-based)',xlim=(0,19),ylim=(0,102),xticks=[0,5,10,15,19]);ax.grid(alpha=.2)
axs[0].set_ylabel('Top-tile overlap (%)');fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs,label='Reuse start step',ticks=[0,3,6,9,12,15,18],shrink=.77)
fig.suptitle('Q64 × K64 softmax tile reuse vs. original DraftMap | H3 layer 25',fontsize=14)
fig.legend(handles=[Line2D([0],[0],color='black',ls='--',lw=2.5,label='DraftMap vs. true, same step'),Line2D([0],[0],color='#328eaa',lw=1.6,label='True map reused from start step')],loc='outside lower center',ncol=2,frameon=False)
fig.savefig(out/'forward_overlap_q64_with_draft.png',dpi=180,bbox_inches='tight');fig.savefig(out/'forward_overlap_q64_with_draft.pdf',bbox_inches='tight')
(out/'overlap.json').write_text(json.dumps({'status':'complete','block':64,'blocks_per_axis':n,'prefix_tokens':prefix,'heads':heads,'layer':25,'prompt_id':16,'seed':20026,'method':'per-head global top-k intersection / k; true tile means FP32, softmax includes prefix; DraftMap maxpool_weight=0.2','rates':results},indent=2)+'\n')
print('complete blocks',n)
