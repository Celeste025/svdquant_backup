#!/usr/bin/env python3
"""Matched full attention API latency and error on E142 captured H3 QKV."""
import json
from pathlib import Path
import statistics
import torch
import bench_e142_attention_tradeoff as b
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.routing import _route_counts

@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    results = {}
    for sample in ('race_s00_b00', 'race_s10_b25', 'clap_s10_b49'):
        p = torch.load(b.DATA / f'{sample}.pt', map_location='cpu', weights_only=True, mmap=True)
        q, k, v = [p[n].cuda().contiguous() for n in ('q','k','v')]
        qb, kb, vb = [x.transpose(1,2).contiguous() for x in (q,k,v)]
        layout = b.VisualLayout(tuple(p['video_grid']), p['prefix_tokens'])
        scales=json.loads((b.ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales']
        cal=b.NVFP4Calibration(*(float(scales[n][p['layer']]) for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
        geom=materialize_ragged_2d_layout(q.device,frames=layout.video_shape[0],height=layout.video_shape[1],width=layout.video_shape[2],logical_block=64,enable_anchors=False)
        items=geom.counts.numel() ** 2
        ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
        arms={}
        for name,skip,i in [('int8',.8,1.),('mixed',.743,6.4/25.7)]:
            sparse=b.SparseConfig(query_block_size=64,sparsity_ratio=skip,maxpool_weight=.2)
            quant=b.QuantConfig(nvfp4_ratio=1-i,int8_ratio=i,prefix_kv_precision='int8',prefix_query_precision='int8')
            def call(s=sparse,quant_config=quant):
                return b.anemoi_attention(qb,kb,vb,layout=layout,layer=p['layer'],sparse_config=s,quant_config=quant_config,calibration=cal if quant_config.nvfp4_ratio else None)
            out=call().transpose(1,2)
            assert torch.isfinite(out).all()
            n16,n8,n4=_route_counts(items,sparsity_ratio=skip,nvfp4_ratio=1-i,fp8_ratio=i,fp16_ratio=0.)
            arms[name]={'fn':call,'metrics':b.metrics(out,ref,p['prefix_tokens']),'actual_pair_fractions':{'skip':1-(n8+n4)/items,'fp4':n4/items,'int8':n8/items},'timings':[]}
        for order in (('int8','mixed'),('mixed','int8')):
            for name in order:
                arms[name]['timings'].append(b.measure(arms[name]['fn'],lambda:None))
        for row in arms.values():
            del row['fn']
            row['mean_ms']=statistics.mean(t['mean_ms'] for t in row['timings'])
        results[sample]=arms
        print(sample, {n:r['mean_ms'] for n,r in arms.items()}, flush=True)
    target=b.ROOT/'results/research/E147/operator_benchmark.json'
    target.write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__': main()
