#!/usr/bin/env python3
"""Check the large tail-layer error against the non-optimized Sage3 path."""
import json
from pathlib import Path
import torch
import bench_e142_attention_tradeoff as bench


@torch.inference_mode()
def main():
    rows=[]
    for sample in ['race_s00_b00','race_s10_b25','clap_s10_b49']:
        packet=torch.load(bench.DATA/f'{sample}.pt',map_location='cpu',weights_only=True,mmap=True)
        q,k,v=[packet[n].cuda().contiguous() for n in ['q','k','v']]
        originals=[t.clone() for t in [q,k,v]]
        ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
        normal=bench.sageattn3_blackwell(q,k.clone(),v,is_causal=False,per_block_mean=True)
        optimized=bench.sageattn3_blackwell(q,k,v,is_causal=False,per_block_mean=True,
                                          fast_preprocess=True,fast_qpack=True,
                                          fast_delta_gemm=True,fast_raw_k_delta=True)
        assert all(torch.equal(a,b) for a,b in zip([q,k,v],originals)), 'Sage3 changed input'
        stats=bench.metrics(optimized,ref,packet['prefix_tokens'])
        gap=float((normal.float()-optimized.float()).square().sum()/normal.float().square().sum())
        assert gap<1e-8,(sample,gap)
        expected=json.loads((bench.REPORT/f'{sample}.json').read_text())['errors']['sage3_optimized']['video']['nmse']
        assert abs(stats['video']['nmse']-expected)<1e-8
        rows.append({'sample':sample,'video_nmse':stats['video']['nmse'],
                     'unoptimized_vs_optimized_nmse':gap,'input_unchanged':True})
        del q,k,v,originals,ref,normal,optimized
    (bench.REPORT/'sage3_validation.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
