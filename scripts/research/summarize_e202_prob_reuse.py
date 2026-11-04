#!/usr/bin/env python3
"""Summarize four case matched E202 videos with original controls."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/E202';ids=(25,48,44,7);modes=('first','periodic','baseline')
labels={'first':'首步捕获概率 map 后复用','periodic':'每 5 步刷新概率 map','baseline':'Baseline：每步每层 DraftMap'}
old=json.loads((ROOT/'results/research/E161/summary.json').read_text());assert old['status']=='complete'
results=[]
for pid in ids:
 data=json.loads((out/f'p{pid:03d}_metrics.json').read_text());assert data['status']=='complete';oldrow=next(r for r in old['rows'] if r['prompt_id']==pid);oldbase=json.loads((ROOT/f'results/research/E162/p{pid:03d}.json').read_text())['metrics']['proxy_new']
 for mode in modes:
  r=data['policies'][mode];old_lp=oldrow['first_only_lpips'] if mode=='first' else oldrow['periodic_lpips'] if mode=='periodic' else oldbase['lpips_alex'];old_ps=oldrow['first_only_psnr'] if mode=='first' else oldrow['periodic_psnr'] if mode=='periodic' else oldbase['psnr_db']
  results.append({'prompt_id':pid,'frames':data['frames'],'policy':mode,'lpips':r['lpips_alex'],'psnr_db':r['psnr_db'],'dit_total_s':r['dit_seconds'],'dit_step_mean_s':r['dit_seconds']/20,'old_lpips':old_lp,'old_psnr_db':old_ps,'video':r['video']})
summary=[]
for mode in modes:
 group=[r for r in results if r['policy']==mode]
 def avg(key,items=group):return statistics.mean(r[key] for r in items)
 summary.append({'policy':mode,'lpips_mean_4':avg('lpips'),'psnr_mean_4':avg('psnr_db'),'dit_step_mean_4_s':avg('dit_step_mean_s'),'dit_step_mean_5s_s':avg('dit_step_mean_s',[r for r in group if r['frames']==124]),'dit_step_mean_10s_s':avg('dit_step_mean_s',[r for r in group if r['frames']==243]),'dit_step_p025_s':next(r['dit_step_mean_s'] for r in group if r['prompt_id']==25),'old_lpips_mean_4':avg('old_lpips'),'old_psnr_mean_4':avg('old_psnr_db')})
payload={'status':'complete','settings':'E150 BF16 reference; 768×1344; 20 steps; same seed and initial audio/video noise; 4 videos equally weighted','timing':'wall-clock DiT forward synchronized, model load/decode excluded; first and periodic use two-pass probability capture; baseline uses per-step DraftMap','rows':results,'summary':summary};(out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
with (out/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
lines=['# E202 首步 softmax 概率 map 复用：四视频对照','','同 E150 的 4 条 BF16 参考：猫麻将和划艇 124 帧，游泳和宇航员 243 帧；视频各占四分之一。LPIPS 对全部帧取平均，PSNR 使用全部帧 RGB 全局 MSE。','', '| 方案 | 四视频平均 LPIPS ↓ | 四视频平均 PSNR ↑ | 5 秒 / 10 秒平均每 step（秒） | 小猫麻将每 step（秒） |','|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {labels[r['policy']]} | {r['lpips_mean_4']:.4f} | {r['psnr_mean_4']:.2f} | {r['dit_step_mean_5s_s']:.2f} / {r['dit_step_mean_10s_s']:.2f} | {r['dit_step_p025_s']:.2f} |")
lines+=['','## 逐视频','','| prompt | 帧 | 方案 | LPIPS | PSNR | DiT 总秒数 | 每 step 秒数 |','|---:|---:|---|---:|---:|---:|---:|']
for r in results:lines.append(f"| {r['prompt_id']} | {r['frames']} | {labels[r['policy']]} | {r['lpips']:.4f} | {r['psnr_db']:.2f} | {r['dit_total_s']:.2f} | {r['dit_step_mean_s']:.2f} |")
lines+=['','首步/周期策略：首层 BF16，首步或每五步 dense INT8 并捕获 map，其他步骤 90% NVFP4 + 10% INT8；baseline 全 step、全层 DraftMap，线性层 BF16。新概率 map 采用额外两遍 INT8 QK 的实验实现，不是融合算子。计时不包括模型加载和解码；GPU 上存在其他低利用率常驻进程，绝对时延仅作本次测量参考。旧策略画质逐 case 数据和原始每 step 时间见 summary.json 与各 policy 日志。','']
(out/'README.md').write_text('\n'.join(lines));print('\n'.join(lines[:11]),flush=True)
