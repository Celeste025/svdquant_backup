#!/usr/bin/env python3
"""Add same-step DraftMap versus true top-tile recall to forward overlap plots."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/tile_probability_overlap_20261103'
true=np.load(out/'step_maps.npz')['maps'];draft=np.load(out/'draft_maps.npz')['maps'];assert true.shape==draft.shape==(20,56,148,148)
steps,heads,n,_=true.shape;total=n*n;mat=json.loads((out/'forward_step_overlap.json').read_text());rates=(.1,.2,.3);results={}
fig,axs=plt.subplots(1,3,figsize=(15,5.3),sharey=True,layout='constrained');cmap=plt.get_cmap('turbo');norm=Normalize(0,18)
for ax,rate in zip(axs,rates):
 k=math.floor(rate*total+.5); t=np.argsort(-true.reshape(steps,heads,-1),axis=-1,kind='stable')[:,:,:k];d=np.argsort(-draft.reshape(steps,heads,-1),axis=-1,kind='stable')[:,:,:k]
 mask=np.zeros((steps,heads,total),dtype=bool);np.put_along_axis(mask,t,True,axis=-1);recall=np.take_along_axis(mask,d,axis=-1).mean(axis=-1)
 results[str(rate)]={'tiles_per_head':k,'mean_by_step':recall.mean(axis=-1).tolist(),'overlap_by_step_head':recall.tolist()}
 m=np.array(mat['rates'][str(rate)]['mean_by_start_and_later_step'])
 for start in range(19):ax.plot(np.arange(start,steps),m[start,start:]*100,color=cmap(norm(start)),linewidth=1.35,alpha=.72)
 ax.plot(np.arange(steps),recall.mean(axis=-1)*100,color='black',linestyle='--',linewidth=2.6,zorder=5)
 ax.set(title=f'Top {int(rate*100)}% tiles',xlabel='Later / current step (zero-based)',xlim=(0,19),ylim=(0,102),xticks=[0,5,10,15,19]);ax.grid(alpha=.2)
axs[0].set_ylabel('Top-tile overlap (%)')
fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs,label='Reuse start step',ticks=[0,3,6,9,12,15,18],shrink=.77)
fig.suptitle('Softmax tile reuse vs. original DraftMap | H3 layer 25',fontsize=14)
fig.legend(handles=[Line2D([0],[0],color='black',ls='--',lw=2.5,label='DraftMap vs. true, same step'),Line2D([0],[0],color='#328eaa',lw=1.6,label='True map reused from start step')],loc='outside lower center',ncol=2,frameon=False)
fig.savefig(out/'forward_step_overlap_with_draft.png',dpi=180,bbox_inches='tight');fig.savefig(out/'forward_step_overlap_with_draft.pdf',bbox_inches='tight')
(out/'draft_same_step_overlap.json').write_text(json.dumps({'status':'complete','method':'Same-step original DraftAttn2 DraftMap versus FP32 softmax mean; matching head global video tile top-k, intersection / k','maxpool_weight':.2,'layer':25,'rates':results},indent=2)+'\n')
print('complete')
