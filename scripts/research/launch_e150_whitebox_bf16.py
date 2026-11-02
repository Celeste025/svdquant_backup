#!/usr/bin/env python3
"""Run E150 GPU shard independently; preserve failures and resume completed phases."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results/research/E150'
WORKER=ROOT/'scripts/research/run_e150_whitebox_bf16.py'

def wait_idle(gpu):
    while True:
        values=[]
        for _ in range(3):
            out=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            used,util=map(int,out.strip().split(','));values.append((used,util))
            time.sleep(10)
        if all(used<1024 and util<10 for used,util in values):return
        print(f'WAIT gpu={gpu} recent={values}',flush=True)
        time.sleep(120)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--wait-idle',action='store_true');args=ap.parse_args()
    manifest=json.loads((REPORT/'manifest.json').read_text());group=next(g for g in manifest['groups'] if g['gpu']==args.gpu)
    if args.wait_idle:wait_idle(args.gpu)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),DIFFSYNTH_SKIP_DOWNLOAD='True',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        MINIMAX_H3_MODEL_ROOT='/home/wjq/workspace/DiffSynth-Studio/models/MiniMax/MiniMax-H3',
        MINIMAX_H3_DIT_PATH='/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors',
        MINIMAX_H3_PROMPT_FILE=manifest['prompt_source'])
    for pid in group['cases']:
        folder=REPORT/f'p{pid:03d}'; denoise=folder/'bf16_denoise.json';video=folder/'bf16_video.json'
        if video.exists() and json.loads(video.read_text()).get('status')=='complete':
            print(f'SKIP p{pid:03d}: complete',flush=True);continue
        if shutil.disk_usage('/data1').free<20*1024**3:
            print('STOP disk free below 20 GiB',flush=True);sys.exit(2)
        if denoise.exists():
            status=json.loads(denoise.read_text()).get('status')
            if status!='denoise_complete':
                print(f'SKIP p{pid:03d}: incomplete denoise status={status}; preserved for inspection',flush=True);continue
        else:
            print(f'START p{pid:03d} denoise at {time.strftime("%F %T")}',flush=True)
            r=subprocess.run([sys.executable,str(WORKER),'--phase','denoise','--prompt-id',str(pid)],env=env)
            if r.returncode:
                print(f'FAILED p{pid:03d} denoise exit={r.returncode}; continue shard',flush=True);continue
        print(f'START p{pid:03d} decode at {time.strftime("%F %T")}',flush=True)
        r=subprocess.run([sys.executable,str(WORKER),'--phase','decode','--prompt-id',str(pid)],env=env)
        if r.returncode:print(f'FAILED p{pid:03d} decode exit={r.returncode}; preserved',flush=True)
    print(f'SHARD_COMPLETE gpu={args.gpu}',flush=True)

if __name__=='__main__':main()
