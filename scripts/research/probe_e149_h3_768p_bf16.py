#!/usr/bin/env python3
"""Time three real H3 BF16 denoising forwards at 768p, then stop."""
import argparse
import importlib
import json
from pathlib import Path
import time
import traceback
import types
import torch
import run_e121_h3_split_generation as e121
from minimax_h3_svdquant_common import read_prompt
attention = importlib.import_module('diffsynth.core.attention.attention')

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/research/E149'

class Complete(Exception):
    pass

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frames',type=int,choices=(124,362),required=True)
    args=ap.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    attention.ATTENTION_IMPLEMENTATION='torch'
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f'bf16_768p_{args.frames}f.json'
    assert not target.exists()
    row=read_prompt(16)
    report={'status':'running','prompt_id':16,'seed':int(row['seed']),
            'settings':{'height':768,'width':1344,'frames':args.frames,'steps':20,'cfg_scale':1.0,'flow_shift':12.0,'audio_flow_shift':3.0},
            'gpu':torch.cuda.get_device_name(),'attention_backend':'torch BF16 SDPA',
            'timing_scope':'synchronized DiT forward including weight offload; no model load or VAE', 'steps':[]}
    e121.base.save_json(target,report)
    start=time.monotonic()
    router=None
    try:
        pipe=e121.base.load_h3_pipeline(full=True,vram_limit_gib=18.0)
        pipe.load_models_to_device(['dit'])
        pipe.dit.eval()
        report['linears']=e121.base.checked_linears(pipe.dit)
        router=e121.Router(pipe.dit,'bf16')
        orig=pipe.dit.forward
        def forward(_self,*a,**kw):
            contract=e121.base.attention_contract(kw)
            torch.cuda.synchronize()
            t=time.monotonic()
            with router.forward_context(**contract,diagnostics=False) as log:
                result=orig(*a,**kw)
                torch.cuda.synchronize()
            elapsed=time.monotonic()-t
            v,audio=e121.base.split_output(result)
            assert torch.isfinite(v).all() and torch.isfinite(audio).all()
            report['steps'].append({'step':len(report['steps']),'dit_seconds':elapsed,
               'main_tokens':contract['expected_cu'][1], 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
               'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
               'free_gib':torch.cuda.mem_get_info()[0]/1024**3,
               'attention_main_calls':log.summary['main_helper_calls']})
            e121.base.save_json(target,report)
            print(json.dumps(report['steps'][-1]),flush=True)
            if len(report['steps'])==3: raise Complete()
            return result
        pipe.dit.forward=types.MethodType(forward,pipe.dit)
        try:
            pipe(prompt=row['prompt'],seed=int(row['seed']),height=768,width=1344,
                 num_frames=args.frames,num_inference_steps=20,cfg_scale=1.0,
                 flow_shift=12.0,audio_flow_shift=3.0,rand_device='cpu',tiled=True)
        except Complete:
            report['status']='complete_three_forwards'
        else:
            raise RuntimeError('expected early stop')
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
        raise
    finally:
        report['total_elapsed_seconds']=time.monotonic()-start
        e121.base.save_json(target,report)
        if router is not None: router.close()

if __name__=='__main__': main()
