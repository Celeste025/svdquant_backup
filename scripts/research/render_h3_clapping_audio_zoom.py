#!/usr/bin/env python3
"""Render only the frozen E042 arrays with fixed margins; no signal recomputation."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/research/E042/readout.json'
OUT = ROOT/'results/research/E042/render_zoom_v2.json'
assert not OUT.exists()
def record(p):
    p=Path(p)
    return dict(file=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
d=json.loads(SOURCE.read_text())
assert d['status']=='complete'
report=dict(status='complete',source=record(__file__),readout=record(SOURCE),
            purpose='Fixed-margin zoom rendering only; first automatic-layout images retained.',
            model_calls=0,gpu_calls=0,signal_recomputations=0,plots=[])
for replica in (0,1):
    fig,axes=plt.subplots(3,1,figsize=(16,9),sharex=True)
    fig.subplots_adjust(left=.075,right=.99,bottom=.16,top=.87,hspace=.31)
    rows=[r for r in d['cases'] if r['replica']==replica]
    for ax,row in zip(axes,rows,strict=True):
        path=Path(row['arrays']['file']); assert record(path)==row['arrays']
        with np.load(path) as a:
            for x in row['visual']['uncountable_or_incomplete']:
                lo,hi=x['seconds']
                if lo<1.4 and hi>0: ax.axvspan(max(0,lo),min(1.4,hi),color='#9d9d9d',alpha=.2)
            for x in row['visual']['closure_intervals']:
                lo,hi=x['seconds']
                if lo<1.4 and hi>0: ax.axvspan(max(0,lo),min(1.4,hi),color='#2ca02c',alpha=.13)
            for t,y,label,color,lw in [
                ('rms_time_seconds','rms_normalized','Stereo RMS (10ms)','#1f77b4',1.15),
                ('hf_time_seconds','hf_sqrt_mean_power_normalized','HF energy (20ms Hann, >=1kHz)','#ff7f0e',.9),
                ('hf_time_seconds','hf_positive_first_difference_normalized','Positive HF difference','#9467bd',.75)]:
                ax.plot(a[t],a[y],label=label,color=color,lw=lw)
        ax.set(xlim=(0,1.4),ylim=(-.02,1.06),ylabel='Own-max normalized')
        ax.set_title(f"{row['case_id']} / {row['arm']} (own E041 intervals)",loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.22)
    axes[-1].set_xlabel('Time from original zero start (seconds)')
    handles,labels=axes[0].get_legend_handles_labels()
    handles += [Patch(facecolor='#2ca02c',alpha=.25),Patch(facecolor='#9d9d9d',alpha=.35)]
    labels += ['E041 closure appearance (contact uncertain)','E041 uncountable / incomplete']
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,fontsize=9)
    fig.suptitle('E042 fixed 0-1.4s zoom: energy readout, not clap detections or sync scores\nEach curve uses its OWN full-track maximum; do not compare loudness. Unannotated does not mean no contact.',y=.97,fontsize=12)
    target=path.parent/f'vbench0161_r{replica}_zoom_0_1p4_layout_v2.png'
    assert not target.exists()
    fig.savefig(target,dpi=150);plt.close(fig)
    report['plots'].append(dict(replica=replica,range_seconds=[0,1.4],artifact=record(target)))
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(OUT)
