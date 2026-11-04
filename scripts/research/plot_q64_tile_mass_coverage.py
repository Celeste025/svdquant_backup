#!/usr/bin/env python3
"""Probability mass captured by historical Q64 top-mean selections."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/tile_probability_coverage_q64_20261103';tops=ROOT/'results/research/tile_probability_overlap_q64_20261103'
ms=[];ps=[];true=[];draft=[];counts0=None
for s in range(20):
 with np.load(out/f'mass_step_{s:02d}.npz') as f:
  ms.append(f['mass'].reshape(56,-1));ps.append(f['prefix_mass'].sum(-1));counts=f['counts']
  if counts0 is None:counts0=counts
  else:assert np.array_equal(counts0,counts)
 with np.load(tops/f'top_step_{s:02d}.npz') as f:true.append(f['true']);draft.append(f['draft'])
mass=np.stack(ms);prefix=np.stack(ps);true=np.stack(true);draft=np.stack(draft);steps,heads,total=mass.shape
video=mass.sum(-1);allkeys=video+prefix
assert np.allclose(allkeys,counts0.sum(),rtol=2e-5)
rates=(.1,.2,.3);results={}
for rate in rates:
 k=math.floor(rate*total+.5);cov=np.empty((steps,steps,heads),np.float32);base=np.empty((steps,heads),np.float32)
 for t in range(steps):
  for start in range(steps):cov[start,t]=np.take_along_axis(mass[t],true[start,:,:k],axis=1).sum(-1)
  base[t]=np.take_along_axis(mass[t],draft[t,:,:k],axis=1).sum(-1)
 values={}
 for label,den in [('video',video),('all_keys',allkeys)]:
  a=cov/den[None,:,:];b=base/den
  assert np.all((a>=0)&(a<=1.00001)) and np.all((b>=0)&(b<=1.00001))
  values[label]={'reuse_by_start_current_head':a.tolist(),'reuse_mean_by_start_current':a.mean(-1).tolist(),'draft_by_current_head':b.tolist(),'draft_mean_by_current':b.mean(-1).tolist()}
 results[str(rate)]={'tiles_per_head':k,'coverage':values}
summary={'status':'complete','definition':'selected video tile softmax probability sum across all video query tokens and video keys, divided per head by video-key mass or all-key mass (video + prefix); average per-head ratios','note':'top selection uses mean per valid query-key pair; probability coverage is sum over all valid query-key pairs; prefix query excluded','layer_zero_based':25,'block':64,'heads':heads,'prompt_id':16,'seed':20026,'video_tokens':int(counts0.sum()),'rates':results,'video_key_fraction_by_step_head':(video/allkeys).tolist()}
(out/'coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
for label,title in [('video','Share of video-key attention probability captured'),('all_keys','Share of all-key attention probability captured')]:
 fig,axs=plt.subplots(1,3,figsize=(15,5.3),sharey=True,layout='constrained');cmap=plt.get_cmap('turbo');norm=Normalize(0,18)
 for ax,rate in zip(axs,rates):
  data=results[str(rate)]['coverage'][label];m=np.array(data['reuse_mean_by_start_current']);baseline=data['draft_mean_by_current']
  for start in range(19):ax.plot(np.arange(start,steps),m[start,start:]*100,color=cmap(norm(start)),linewidth=1.35,alpha=.72)
  ax.plot(np.arange(steps),np.array(baseline)*100,color='black',linestyle='--',linewidth=2.6,zorder=5)
  ax.set(title=f'Top {int(rate*100)}% tiles',xlabel='Later / current step (zero-based)',xlim=(0,19),ylim=(0,102),xticks=[0,5,10,15,19]);ax.grid(alpha=.2)
 axs[0].set_ylabel('Probability mass captured (%)');fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axs,label='Reuse start step',ticks=[0,3,6,9,12,15,18],shrink=.77)
 fig.suptitle(f'{title} | Q64 × K64, H3 layer 25',fontsize=13)
 fig.legend(handles=[Line2D([0],[0],color='black',ls='--',lw=2.5,label='DraftMap, same step'),Line2D([0],[0],color='#328eaa',lw=1.6,label='True top-mean map reused from start step')],loc='outside lower center',ncol=2,frameon=False)
 fig.savefig(out/f'coverage_{label}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'coverage_{label}.pdf',bbox_inches='tight');plt.close(fig)
print('complete')
