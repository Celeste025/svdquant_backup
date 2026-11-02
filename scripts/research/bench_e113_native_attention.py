#!/usr/bin/env python3
"""Full real-H3 attention check for native NVFP4/FP8 mixed Triton."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research'))
from e113_native_mixed_attention import native_mixed_sage3_attention
from sageattn3 import sageattn3_blackwell


def nmse(a,b):
    a,b=a.float(),b.float()
    return ((a-b).square().sum()/b.square().sum()).item()


def run(name,fn):
    warm=fn()
    torch.cuda.synchronize()
    assert torch.isfinite(warm).all().item()
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    samples=[]
    for _ in range(3):
        start.record()
        out=fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    assert torch.isfinite(out).all().item()
    return out,sum(samples)/len(samples)


def main():
    capture=Path('/data1/models/svdquant-wjq/research/20261014/E098/h3_p1_s0_b25_480x832_73f_qkv.pt')
    payload=torch.load(capture,map_location='cpu',weights_only=True,mmap=True)
    q,k,v=(payload[key][:9482].transpose(0,1).unsqueeze(0).contiguous().cuda()
           for key in ('q','k','v'))
    with torch.inference_mode():
        native,t_native=run('sage3',lambda:sageattn3_blackwell(q,k.clone(),v,is_causal=False,per_block_mean=True))
        all4,t_all4=run('all4',lambda:native_mixed_sage3_attention(q,k,v,threshold=.5))
        mixed,t_mixed=run('mixed',lambda:native_mixed_sage3_attention(q,k,v,threshold=.1))
        bf16,t_bf16=run('bf16',lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v))
        row={'shape':list(q.shape),'gpu':torch.cuda.get_device_name(),
             'time_ms':{'sage3':t_native,'native_triton_all_fp4':t_all4,
                        'native_triton_mixed':t_mixed,'bf16':t_bf16},
             'nmse_vs_bf16':{'sage3':nmse(native,bf16),'native_triton_all_fp4':nmse(all4,bf16),
                             'native_triton_mixed':nmse(mixed,bf16)},
             'all_fp4_vs_sage3_nmse':nmse(all4,native),
             'mixed_vs_all_fp4_nmse':nmse(mixed,all4),
             'mixed_vs_sage3_nmse':nmse(mixed,native),
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    output=ROOT/'results/research/E113/native_full_attention_preflight.json'
    output.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,indent=2),flush=True)


if __name__=='__main__':
    main()
