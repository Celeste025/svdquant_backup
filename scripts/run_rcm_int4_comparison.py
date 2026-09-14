#!/usr/bin/env python3
"""Generate the fixed BF16/NVFP4/INT4 rCM-Wan comparison suite and score latents."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]; DATA=Path('/data1/models/svdquant-wjq')
FAST='A professional skateboarder performing a high-speed kickflip down a concrete stair set, dynamic tracking shot.'
SEEN=(0,1,6,8,15); UNSEEN=(57,67,71,98,121)

def metric(a,b):
 a=a.float(); b=b.float(); d=a-b
 return {'mse':float(d.square().mean()),'nmse':float(d.square().sum()/b.square().sum().clamp_min(1e-30)),'cosine':float((a.flatten()@b.flatten())/(a.flatten().norm()*b.flatten().norm()).clamp_min(1e-30))}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results/samples/rcm_int4_svdquant_g10'); p.add_argument('--gpu',default='6'); p.add_argument('--skip-existing',action='store_true'); a=p.parse_args()
 prompts=yaml.safe_load((ROOT/'third_party/deepcompressor/examples/diffusion/prompts/vbench_t2v_simple.yaml').read_text())
 cases=[{'id':f'fastmotion_seed{s}','prompt':FAST,'seed':s,'split':'fastmotion'} for s in (308,309)]
 for split,ids in [('seen',SEEN),('unseen',UNSEEN)]:
  for i in ids: cases.append({'id':f'{i:04d}','prompt':prompts[f'{i:04d}'],'seed':10000+i,'split':split,'vbench_id':i})
 models={'bf16':None,'nvfp4':DATA/'ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16','int4':DATA/'ckpts/rcm-wan2.1-1.3b-int4-s16-g10'}
 for case in cases:
  d=a.output/case['id']; d.mkdir(parents=True,exist_ok=True)
  for tag,ckpt in models.items():
   latent=d/f'{tag}.pt'; video=d/f'{tag}.mp4'
   if a.skip_existing and latent.exists() and video.exists(): continue
   cmd=[sys.executable,str(ROOT/'scripts/infer_rcm_wan_4step.py'),'--model',str(DATA/'models/Wan2.1-T2V-1.3B-Diffusers'),'--transformer',str(DATA/'models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer'),'--prompt',case['prompt'],'--seed',str(case['seed']),'--height','480','--width','832','--frames','77','--steps','4','--output',str(video),'--latent-output',str(latent)]
   if ckpt: cmd += ['--quant-ckpt',str(ckpt), '--quant-recipe','int4' if tag=='int4' else 'real-nvfp4']
   subprocess.run(cmd,check=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':a.gpu})
  bf=torch.load(d/'bf16.pt',map_location='cpu',weights_only=True)
  case['metrics']={tag:metric(torch.load(d/f'{tag}.pt',map_location='cpu',weights_only=True),bf) for tag in ('nvfp4','int4')}
  (d/'case.json').write_text(json.dumps(case,indent=2)+'\n')
 a.output.mkdir(parents=True,exist_ok=True); (a.output/'summary.json').write_text(json.dumps(cases,indent=2)+'\n')
if __name__=='__main__': main()
