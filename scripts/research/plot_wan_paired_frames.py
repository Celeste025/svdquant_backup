#!/usr/bin/env python3
"""Frame contact sheet for the one E008 paired video; no metric/quality claim."""
from pathlib import Path
import json
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
d=json.loads((ROOT/'results/research/E008_wan_native_paired_video.json').read_text())
assert d['status']=='complete'
indices=[0,25,50,76]
fig,axes=plt.subplots(2,4,figsize=(16,5.5),layout='constrained')
for row,arm in enumerate(['bf16','nativefast']):
 reader=imageio.get_reader(d['arms'][arm]['video']['path'],'ffmpeg')
 for col,index in enumerate(indices):
  axes[row,col].imshow(reader.get_data(index));axes[row,col].axis('off')
  axes[row,col].set_title(f'{"BF16" if row==0 else "Native NVFP4"} · frame {index}',fontsize=11)
 reader.close()
fig.suptitle('E008 · one calibration prompt: "A person is roller skating" · identical seed/noise · 4-step free rollout',fontsize=12)
p=ROOT/'research_state/reports/figures/E008_paired_frames.png';p.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(p,dpi=140);print(p)
