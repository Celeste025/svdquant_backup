#!/usr/bin/env python3
"""Paired real-QKV end-to-end attention timing and BF16 error."""
import json
from pathlib import Path
import statistics
import torch
import bench_e142_attention_tradeoff as b

DATA=Path('/data1/models/svdquant-wjq/research/20261101/E151_long_h3_attention')
OUT=b.ROOT/'results/research/E151'

@torch.inference_mode()
def main():
    torch.set_num_threads(6);OUT.mkdir(parents=True,exist_ok=True)
    scales=json.loads((b.ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales']
    result={'device':torch.cuda.get_device_name(),'policy':'full attention API; HND to BSHD contiguous performed before timing; prefix INT8; Q64','samples':{}}
    for frames in (124,362):
        p=torch.load(DATA/f'race_768p_{frames}f_step0_block0.pt',map_location='cpu',weights_only=True,mmap=True)
        q,k,v=[p[n].cuda().contiguous() for n in ('q','k','v')];qb,kb,vb=[x.transpose(1,2).contiguous() for x in (q,k,v)]
        layout=b.VisualLayout(tuple(p['video_grid']),p['prefix_tokens']);cal=b.NVFP4Calibration(*(float(scales[k][0]) for k in ['q_tensor_scale','k_tensor_scale','v_tensor_scale']))
        arms={}
        for name,skip,fp4 in [('skip80_int8_20',.8,0.),('skip75_fp4_15_int8_10',.75,.6)]:
            sparse=b.SparseConfig(query_block_size=64,sparsity_ratio=skip,maxpool_weight=.2);quant=b.QuantConfig(nvfp4_ratio=fp4,int8_ratio=1-fp4,prefix_kv_precision='int8',prefix_query_precision='int8')
            def call(s=sparse,qu=quant):return b.anemoi_attention(qb,kb,vb,layout=layout,layer=0,sparse_config=s,quant_config=qu,calibration=cal if qu.nvfp4_ratio else None)
            arms[name]={'fn':call,'rounds':[]}
        for order in (tuple(arms),tuple(reversed(arms))):
            for name in order:
                row=b.measure(arms[name]['fn'],lambda:None);arms[name]['rounds'].append(row);print(frames,name,row['mean_ms'],flush=True)
        ref=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
        for name,row in arms.items():
            output=row.pop('fn')().transpose(1,2);assert torch.isfinite(output).all();row['metrics']=b.metrics(output,ref,p['prefix_tokens']);row['mean_ms']=statistics.mean(x['mean_ms'] for x in row['rounds'])
        result['samples'][str(frames)]={'tokens':q.shape[2],'grid':p['video_grid'],'prefix_tokens':p['prefix_tokens'],'arms':arms}
        (OUT/'paired_long_attention.json').write_text(json.dumps(result,indent=2)+'\n')
        print('DONE',frames,flush=True)

if __name__=='__main__':main()
