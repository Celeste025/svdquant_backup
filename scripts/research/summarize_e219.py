#!/usr/bin/env python3
import csv,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'results/research/E219';rows=[];cases=[]
for policy in ('protect128','protect64'):
 data=[json.loads((out/policy/f'p{pid:03d}'/'metrics.json').read_text()) for pid in (25,48,44,7)]
 assert all(d['status']=='complete' for d in data)
 cases.extend(data)
 rows.append({'policy':policy,'lpips':statistics.mean(d['lpips_alex'] for d in data),'psnr_db':statistics.mean(d['psnr_db'] for d in data),'p025_step_seconds':data[0]['dit_seconds']/20})
(out/'summary.json').write_text(json.dumps({'status':'complete','rows':rows,'cases':cases,'note':'four videos equally weighted; p025 single-run DiT time'},indent=2)+'\n')
lines=['# E219 四视频对照','','四视频等权平均；耗时是 p025 单次 20-step DiT 平均。对照与口径见 E202，捕获步骤的额外计算见 plan.md。','','| 方案 | LPIPS | PSNR (dB) | p025 秒/step |','|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['policy']} | {r['lpips']:.4f} | {r['psnr_db']:.2f} | {r['p025_step_seconds']:.2f} |")
(out/'summary.md').write_text('\n'.join(lines)+'\n')
