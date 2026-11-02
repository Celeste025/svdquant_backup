#!/usr/bin/env python3
"""Summarize paired same-input DraftMap and oracle block top-k recalls."""
import itertools,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/research/draftmap_maxprob_20261102';LAYERS=(0,12,24,36,49);RATES=(.1,.2,.3,.5);ORACLES=('mass','qkmax','maxprob')
rows=[json.loads(x) for x in (OUT/'per_cell.jsonl').read_text().splitlines()];assert len(rows)==100 and {(x['step'],x['layer']) for x in rows}=={(t,l) for t in range(20) for l in LAYERS}
arr=np.empty((3,4,20,5,8))
for row in rows:
 for oi,oracle in enumerate(ORACLES):
  for ri,rate in enumerate(RATES):arr[oi,ri,row['step'],LAYERS.index(row['layer'])]=row['budget'][str(rate)]['recall_by_head'][oracle]
rng=np.random.default_rng(20261102);perms=np.stack([rng.permutation(20) for _ in range(50000)]);layer_perms=np.array(list(itertools.permutations(range(5))))
def correlation_permutation(y,permutations,exhaustive):
 y=rankdata(y);y=y-y.mean();x=np.arange(len(y),dtype=float);x-=x.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
 if den==0:return {'rho':None,'p_permutation':None}
 observed=float(x@y/den);values=(y[permutations]@x)/den;count=int(np.sum(np.abs(values)>=abs(observed)-1e-12));p=count/len(values) if exhaustive else (count+1)/(len(values)+1)
 return {'rho':observed,'p_permutation':p,'permutations':len(values)}
def holm(ps):
 valid=sorted([(p,i) for i,p in enumerate(ps) if p is not None]);out=[None]*len(ps);running=0
 for j,(p,i) in enumerate(valid):running=max(running,(len(valid)-j)*p);out[i]=min(1.,running)
 return out
results={}
for oi,oracle in enumerate(ORACLES):
 tests=[];metrics={}
 for ri,rate in enumerate(RATES):
  a=arr[oi,ri];tmean=a.mean(axis=(1,2));lmean=a.mean(axis=(0,2));tt=correlation_permutation(tmean,perms,False);ll=correlation_permutation(lmean,layer_perms,True)
  row={'mean_recall':float(a.mean()),'random_baseline':rate,'min_cell_mean':float(a.mean(axis=2).min()),'max_cell_mean':float(a.mean(axis=2).max()),'step_means':tmean.tolist(),'layer_means':lmean.tolist(),'time_spearman':tt,'depth_spearman':ll};metrics[str(rate)]=row;tests.extend([tt,ll])
 corrected=holm([t['p_permutation'] for t in tests])
 for test,p in zip(tests,corrected):test['p_holm_within_oracle']=p
 results[oracle]=metrics
report={'status':'complete','sample':{'prompt_id':16,'seed':20026,'trajectory':'BF16 dense closed-loop','steps':20,'layers':LAYERS,'heads':[0,8,16,24,32,40,48,55],'block_size':128,'maxpool_weight':.2,'fps_frames':'480x832 / 124 frames','unit':'global per-head top-k video tile; macro mean across 20 steps, 5 layers, 8 heads'},'oracle_definitions':{'mass':'sum of actual softmax probability over valid key tokens, averaged over valid query tokens in each tile; softmax includes prefix keys','qkmax':'maximum valid BF16-matmul QK logit in tile; proxy for exact routing, not the quantized INT8 capture route','maxprob':'maximum FP32 softmax probability over valid query-key token pairs in each video tile, with prefix keys included in normalization'},'statistics':'Spearman on 20 step means or 5 layer means; global label permutation, exact 120 layer permutations and 50000 random timestep permutations; Holm correction over 8 tests per oracle. Exchangeability is an assumption; temporal/depth dependence and one prompt limit inference.','results':results}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(13,13),layout='constrained');colors=['#af3340','#da8d1d','#2478b5','#319367']
for oi,oracle in enumerate(ORACLES):
 for ri,rate in enumerate(RATES):
  a=arr[oi,ri];axes[oi,0].plot(np.arange(1,21),a.mean(axis=(1,2))*100,label=f'Top {rate:.0%}',color=colors[ri]);axes[oi,1].plot(np.array(LAYERS)+1,a.mean(axis=(0,2))*100,marker='o',label=f'Top {rate:.0%}',color=colors[ri])
 for ax in axes[oi]:ax.set_ylim(0,100);ax.grid(alpha=.25);ax.set_ylabel('Top-k overlap recall (%)');ax.legend(ncol=2,fontsize=8)
 axes[oi,0].set_title({'mass':'True softmax block mass','qkmax':'Block max QK logit','maxprob':'Block max softmax probability'}[oracle]+' — by timestep');axes[oi,1].set_title({'mass':'True softmax block mass','qkmax':'Block max QK logit','maxprob':'Block max softmax probability'}[oracle]+' — by layer depth')
 axes[oi,0].set_xlabel('Denoising step (1–20)');axes[oi,1].set_xlabel('Transformer layer (1–50)');axes[oi,1].set_xticks(np.array(LAYERS)+1)
fig.suptitle('DraftAttention2 block selection: same-input top-k recall',fontsize=15)
fig.text(.5,-.01,'One BF16 trajectory, p016; 480×832, 124 frames; Q128, 8 fixed heads, 5 fixed layers. Random-selection expected recall equals retention budget.',ha='center',fontsize=9)
fig.savefig(OUT/'draftmap_hit_by_step_and_layer.png',dpi=170,bbox_inches='tight');fig.savefig(OUT/'draftmap_hit_by_step_and_layer.pdf',bbox_inches='tight')
print(json.dumps({'mass':{k:{'mean':v['mean_recall'],'time':v['time_spearman'],'depth':v['depth_spearman']} for k,v in results['mass'].items()},'maxprob':{k:{'mean':v['mean_recall'],'time':v['time_spearman'],'depth':v['depth_spearman']} for k,v in results['maxprob'].items()}},indent=2))
