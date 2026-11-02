#!/usr/bin/env python3
"""One matched E150 case per route policy using the current Anemoi build."""
import argparse, json, hashlib
from pathlib import Path
import torch
import run_e159_route_reuse_whitebox as first
import run_e161_periodic_route_whitebox as periodic
import run_e162_proxy_new as baseline_router
import run_e150_whitebox_bf16 as run
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=['first','periodic','baseline'],required=True);parser.add_argument('--prompt-id',type=int,default=25);args=parser.parse_args()
lib=ROOT/'third_party/anemoi-code/anemoi/layers/attention/mpa/_cuda_sm120_q64.cpython-312-x86_64-linux-gnu.so'
hashval=hashlib.sha256(lib.read_bytes()).hexdigest();assert hashval=='1f52b6dffc744d3bc175beb53a90d49124e66ee264814aaf39fab766e92949a1',hashval
ref=json.loads((ROOT/f'results/research/E150/p{args.prompt_id:03d}/bf16_denoise.json').read_text())
if args.policy=='first': first.MODE='reuse'; router=first.Router
elif args.policy=='periodic': periodic.MODE='reuse';router=periodic.Router
else: first.MODE='proxy'; router=baseline_router.Router
run.REPORT=ROOT/'results/research/optimized_route_dit'/args.policy
run.DATA=ROOT/'results/local/optimized_route_dit'/args.policy
run.ARM=args.policy;run.e121.Router=router
torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
run.denoise(args.prompt_id)
p=run.REPORT/f'p{args.prompt_id:03d}/bf16_denoise.json';d=json.loads(p.read_text());assert d['initial_latent_sha256']==ref['initial_latent_sha256']
d['benchmark_binary_sha256']=hashval;d['benchmark_policy']=args.policy
p.write_text(json.dumps(d,indent=2)+'\n')
print('TIMING',args.policy,sum(x['dit_seconds'] for x in d['steps']),flush=True)
