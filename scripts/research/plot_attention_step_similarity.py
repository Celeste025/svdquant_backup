#!/usr/bin/env python3
"""Plot Q128 video-only BF16 logit maxima and selected-tile overlap on one head."""
import json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'results/local/attention_step_similarity'
out=ROOT/'results/research/attention_step_similarity'
torch.set_grad_enabled(False)
head=0; tile=128; maps=[]; masks=[]
for step in range(4):
 p=torch.load(base/f'race_480p_124f_step{step}_block25.pt',map_location='cpu',weights_only=False)
 prefix=p['prefix_tokens']; q=p['q'][0,head,prefix:].cuda().float(); k=p['k'][0,head,prefix:].cuda().float(); n=q.shape[0]; count=math.ceil(n/tile)
 # Use valid video tokens only; exclude padding from tile maxima.
 padded=count*tile; k=torch.nn.functional.pad(k,(0,0,0,padded-n)); valid=torch.arange(padded,device='cuda')<n
 scores=[]
 for start in range(0,n,tile):
  logits=(q[start:start+tile]@k.T)*float(p['softmax_scale'])
  logits[:,~valid]=-torch.inf
  scores.append(logits.view(-1,count,tile).amax(dim=(0,2)).cpu().numpy())
 a=np.stack(scores); maps.append(a)
 # Same physical budget for every timestep; ties resolved by stable sorting.
 num=round(a.size*.25); idx=np.argsort(-a.ravel(),kind='stable')[:num]; mask=np.zeros(a.size,dtype=bool); mask[idx]=True; masks.append(mask.reshape(a.shape))
overlap=lambda a,b:float(np.logical_and(a,b).sum()/a.sum())
adj=[overlap(masks[i],masks[i+1]) for i in range(3)]
first=[overlap(masks[0],masks[i]) for i in range(4)]
fig,ax=plt.subplots(2,4,figsize=(16,8),layout='constrained')
lo=min(np.percentile(a,2) for a in maps); hi=max(np.percentile(a,98) for a in maps)
for i in range(4):
 im=ax[0,i].imshow(maps[i],origin='lower',cmap='magma',vmin=lo,vmax=hi,aspect='equal');ax[0,i].set_title(f'Step {i}',fontsize=13);ax[0,i].set_xlabel('Key block index');ax[0,i].set_ylabel('Query block index')
 ax[1,i].imshow(masks[i],origin='lower',cmap='Blues',vmin=0,vmax=1,aspect='equal');ax[1,i].set_title(f'Top 25% blocks | step 0 overlap {first[i]:.1%}',fontsize=10);ax[1,i].set_xlabel('Key block index');ax[1,i].set_ylabel('Query block index')
fig.colorbar(im,ax=list(ax[0,:]),shrink=.85,label='Max scaled QK logit per block')
fig.suptitle('Adjacent denoising steps: attention block scores and selected locations',fontsize=16)
fig.text(.5,-.09,'Race p016, 480x832, layer 25, head 0 | video tokens only | Q128 x K128 | BF16 Q/K, FP32 QK | fixed top 25% per head\nAdjacent top-block overlap: '+', '.join(f'{i}→{i+1}: {v:.1%}' for i,v in enumerate(adj))+' | random-selection reference: 25%',ha='center',fontsize=10)
fig.savefig(out/'adjacent_step_attention_blocks.png',dpi=180,bbox_inches='tight');fig.savefig(out/'adjacent_step_attention_blocks.pdf',bbox_inches='tight')
report={'prompt_id':16,'seed':20026,'resolution':'480x832','frames':124,'layer':25,'head':head,'steps':[0,1,2,3],'tile_size':tile,'video_tokens':n,'grid_shape':list(maps[0].shape),'score':'max scaled QK logit over valid video token pairs, BF16 Q/K with FP32 matmul','selection':'top 25% tiles within selected head; stable sorting','adjacent_overlap_fraction':adj,'step0_overlap_fraction':first,'independent_random_reference':0.25,'limitations':'One head and one prompt; this is a dense BF16 logit-max diagnostic, not native INT8 exact map or official DraftMap; no softmax and no end-to-end reuse guarantee.'}
(out/'adjacent_step_attention_blocks.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out/'adjacent_step_attention_blocks.npz',scores=np.stack(maps),masks=np.stack(masks));print(json.dumps(report,indent=2))
