#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/E215'
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
for ax,(pid,layer,title) in zip(axs.flat,[(25,25,'Mahjong · layer 25'),(25,49,'Mahjong · layer 49'),(48,25,'Kayak · layer 25'),(48,49,'Kayak · layer 49')]):
 d=json.load(open(out/f'p{pid:03d}_step14_layer{layer}.json'));skips=('0.8','0.5','0.0');xx=np.arange(3);width=.19
 for j,(key,label,color) in enumerate([('sparse_nmse','Sparse support','#4477aa'),('quant_nmse','Additional quantization','#ee7733'),('cross_nmse','Cross term','#999999'),('nmse','Total','#cc3311')]):ax.bar(xx+(j-1.5)*width,[100*d['arms'][s][key] for s in skips],width,label=label,color=color)
 ax.axhline(0,color='black',lw=.7);ax.set(title=title,ylabel='NMSE contribution (percentage points)',xticks=xx,xticklabels=['10%','40%','90%'],xlabel='NVFP4 fraction (INT8 fixed at 10%)');ax.grid(axis='y',alpha=.2)
handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False);fig.suptitle('Why can adding NVFP4 increase error? | step 14, Q64 × K64',fontsize=14)
fig.savefig(out/'error_decomposition.png',dpi=160);fig.savefig(out/'error_decomposition.pdf')
