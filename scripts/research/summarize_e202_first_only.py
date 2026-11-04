#!/usr/bin/env python3
"""Add the new first-step softmax route to existing E159–E162 results."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'results/research/E202';ids=(25,48,44,7)
old=json.loads((ROOT/'results/research/E161/summary.json').read_text());assert old['status']=='complete'
new=[]
for pid in ids:
 d=json.loads((out/f'p{pid:03d}_first_metrics.json').read_text());assert d['status']=='complete' and len(d['step_seconds'])==20;new.append(d)
def mean(xs):return statistics.mean(xs)
oldtime={k:sum(s['dit_seconds'] for s in json.loads((ROOT/f'results/research/optimized_route_dit/{k}/p025/bf16_denoise.json').read_text())['steps'])/20 for k in ('first','periodic','baseline')}
base=[json.loads((ROOT/f'results/research/E162/p{pid:03d}.json').read_text())['metrics']['proxy_new'] for pid in ids];rows=[]
for key,label,lp,ps in (
 ('first','原版首步 QK map 复用',mean(r['first_only_lpips'] for r in old['rows']),mean(r['first_only_psnr'] for r in old['rows'])),
 ('periodic','原版每 5 步刷新 QK map',mean(r['periodic_lpips'] for r in old['rows']),mean(r['periodic_psnr'] for r in old['rows'])),
 ('baseline','Baseline 全 step 全层 DraftMap',mean(r['lpips_alex'] for r in base),mean(r['psnr_db'] for r in base))):
 rows.append({'policy':key,'label':label,'lpips':lp,'psnr_db':ps,'time_p025_s':oldtime[key],'source':'existing E160/E161/E162; p025 timing from optimized_route_dit'})
rows.append({'policy':'softmax_first','label':'新增首步 softmax 概率 map 复用','lpips':mean(r['lpips_alex'] for r in new),'psnr_db':mean(r['psnr_db'] for r in new),'time_p025_s':next(r['dit_seconds']/20 for r in new if r['prompt_id']==25),'time_5s_s':mean(r['dit_seconds']/20 for r in new if r['frames']==124),'time_10s_s':mean(r['dit_seconds']/20 for r in new if r['frames']==243),'source':'E202 new generation; two-pass INT8 probability capture'})
payload={'status':'complete','rows':rows,'cases':[{'prompt_id':r['prompt_id'],'frames':r['frames'],'lpips':r['lpips_alex'],'psnr_db':r['psnr_db'],'dit_step_mean_s':r['dit_seconds']/20,'dit_total_s':r['dit_seconds'],'video':r['video']} for r in new],'timing_note':'20-step synchronized DiT wall-clock, model loading/decode excluded; p025 single-run values, 5s/10s new-mode means are reported separately','old_data':'unchanged E160/E161/E162 reports; no new baseline/periodic results used'}
(out/'first_comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
with (out/'first_cases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(payload['cases'][0]));w.writeheader();w.writerows(payload['cases'])
lines=['# E202：新增首步 softmax 概率 map 对照','','旧三组的画质来自 E160/E161/E162 原有结果，耗时为各方案此前在小猫麻将 p025 的单次复测；新增方案只生成首步概率 map 复用，采用两遍 INT8 QK 的实验版算子。各视频与既有 BF16 参考对齐。','','| 方案 | 四视频平均 LPIPS ↓ | 四视频平均 PSNR ↑ | p025 平均每 step（秒） |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['label']} | {r['lpips']:.4f} | {r['psnr_db']:.2f} | {r['time_p025_s']:.2f} |")
r=rows[-1];lines+=['',f"新增方案的平均每 step：5 秒视频 {r['time_5s_s']:.2f} 秒；10 秒视频 {r['time_10s_s']:.2f} 秒。旧表的时间仅为 p025，并非四视频均值。",'','| 案例 | 帧数 | LPIPS | PSNR | DiT 总耗时 | 视频 |','|---:|---:|---:|---:|---:|---|']
for r in payload['cases']:lines.append(f"| {r['prompt_id']} | {r['frames']} | {r['lpips']:.4f} | {r['psnr_db']:.2f} | {r['dit_total_s']:.2f} 秒 | {r['video']} |")
lines+=['','旧数据未覆盖；误启动的 E202 周期方案不用于比较。结果只针对四个视频，不构成统计显著性结论。','']
(out/'first_comparison.md').write_text('\n'.join(lines));print('\n'.join(lines[:11]),flush=True)
