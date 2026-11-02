#!/usr/bin/env python3
"""Same real H3 QKV, full API timing: Draft mixed vs dense Sage2++/Sage3."""
import json
import os
import statistics
from pathlib import Path
import torch
import bench_e142_attention_tradeoff as b

DATA=Path('/data1/models/svdquant-wjq/research/20261101/E151_long_h3_attention')
OUT=b.ROOT/'results/research/E153'

@torch.inference_mode()
def main():
    torch.set_num_threads(6); OUT.mkdir(parents=True,exist_ok=True)
    scales=json.loads((b.ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales']
    report={'gpu':torch.cuda.get_device_name(),'scope':'full API, cached HND and BSHD contiguous, excludes initial layout conversion and data transfer; each round 5 warmup and 3x4 timed calls; order reversed','samples':{}}
    for frames in ([int(os.environ['E153_FRAMES'])] if 'E153_FRAMES' in os.environ else (124,362)):
        p=torch.load(DATA/f'race_768p_{frames}f_step0_block0.pt',map_location='cpu',weights_only=True,mmap=True)
        q,k,v=[p[n].cuda().contiguous() for n in ('q','k','v')];qb,kb,vb=[x.transpose(1,2).contiguous() for x in (q,k,v)]
        layout=b.VisualLayout(tuple(p['video_grid']),p['prefix_tokens'])
        cal=b.NVFP4Calibration(*(float(scales[k][0]) for k in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
        sparse=b.SparseConfig(query_block_size=64,sparsity_ratio=.75,maxpool_weight=.2)
        quant=b.QuantConfig(nvfp4_ratio=.6,int8_ratio=.4,prefix_kv_precision='int8',prefix_query_precision='int8')
        arms={'sage2pp':lambda:b.sageattn(q,k,v,tensor_layout='HND',is_causal=False,sm_scale=128**-.5),
              'sage3':lambda:b.sageattn3_blackwell(q,k,v,is_causal=False,per_block_mean=True,fast_preprocess=True,fast_qpack=True,fast_delta_gemm=True,fast_raw_k_delta=True),
              'draft_mixed':lambda:b.anemoi_attention(qb,kb,vb,layout=layout,layer=0,sparse_config=sparse,quant_config=quant,calibration=cal)}
        if os.environ.get('E153_SKIP_SAGE3') == '1': del arms['sage3']
        result={name:[] for name in arms}
        for order in (list(arms),list(reversed(arms))):
            for name in order:
                for _ in range(5): arms[name]()
                torch.cuda.synchronize(); batches=[]
                for _ in range(3):
                    a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
                    for _ in range(4): out=arms[name]()
                    z.record();z.synchronize();batches.append(a.elapsed_time(z)/4)
                assert torch.isfinite(out).all();del out
                result[name].extend(batches); print(frames,name,batches,flush=True)
        report['samples'][str(frames)]={'tokens':q.shape[2],'prefix_tokens':p['prefix_tokens'],'arms':{name:{'mean_ms':statistics.mean(vals),'batch_means_ms':vals} for name,vals in result.items()}}
        (OUT/('paired_long_safe.json' if os.environ.get('E153_SKIP_SAGE3') == '1' else 'paired.json')).write_text(json.dumps(report,indent=2)+'\n')
        del q,k,v,qb,kb,vb,p
        torch.cuda.empty_cache()

if __name__=='__main__':main()
