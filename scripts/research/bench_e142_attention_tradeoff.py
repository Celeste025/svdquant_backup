#!/usr/bin/env python3
"""Native deployment full-API speed and whole-output error at fixed real QKV."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
SAGE3=Path('/data1/models/svdquant-wjq/research/20261025/E130/optin_rawk_readonly')
SAGE2=Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')
sys.path[:0]=[str(ROOT/'third_party/anemoi-code'),str(SAGE3),str(SAGE2),str(ROOT/'scripts/research')]
from anemoi import anemoi_attention,SparseConfig,QuantConfig,VisualLayout,NVFP4Calibration
from sageattn3 import sageattn3_blackwell
from sageattention import sageattn
from bench_e130_combined_lengths import measure

DATA=Path('/data1/models/svdquant-wjq/research/20261031/E142_attention_tradeoff')
REPORT=ROOT/'results/research/E142'


@torch.inference_mode()
def metrics(out,ref,prefix):
    rows={}
    for label,sl in [('all',slice(None)),('video',slice(prefix,None)),('prefix',slice(None,prefix))]:
        a,b=out[:,:,sl,:].float(),ref[:,:,sl,:].float()
        delta=a-b
        rows[label]={'nmse':float(delta.square().sum(dtype=torch.float64)/b.square().sum(dtype=torch.float64)),
                     'mse':float(delta.square().mean(dtype=torch.float64)),
                     'reference_mean_square':float(b.square().mean(dtype=torch.float64))}
    return rows


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample',required=True)
    sample=ap.parse_args().sample
    torch.set_num_threads(6)
    source=DATA/f'{sample}.pt'
    p=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    q,k,v=[p[n].cuda().contiguous() for n in ['q','k','v']]
    qb,kb,vb=[x.transpose(1,2).contiguous() for x in [q,k,v]]
    prefix=p['prefix_tokens']
    calibration_file=ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json'
    scales=json.loads(calibration_file.read_text())['scales']
    calibration=NVFP4Calibration(*(float(scales[k][p['layer']]) for k in ['q_tensor_scale','k_tensor_scale','v_tensor_scale']))
    layout=VisualLayout(tuple(p['video_grid']),prefix)
    def bf16():
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
    def sage2():
        return sageattn(q,k,v,tensor_layout='HND',is_causal=False,sm_scale=128**-.5)
    def sage3():
        return sageattn3_blackwell(q,k,v,is_causal=False,per_block_mean=True,
                                  fast_preprocess=True,fast_qpack=True,
                                  fast_delta_gemm=True,fast_raw_k_delta=True)
    def make_draft(sparsity,int8_fraction):
        sparse=SparseConfig(query_block_size=64,sparsity_ratio=sparsity,maxpool_weight=.2)
        quant=QuantConfig(nvfp4_ratio=1-int8_fraction,int8_ratio=int8_fraction,
                          fp16_ratio=0.,prefix_kv_precision='int8',prefix_query_precision='int8')
        def call():
            return anemoi_attention(qb,kb,vb,layout=layout,layer=p['layer'],
                                     sparse_config=sparse,quant_config=quant,
                                     calibration=calibration if quant.nvfp4_ratio else None)
        return call
    arms={'bf16_sdpa':bf16,'sage2pp':sage2,'sage3_optimized':sage3}
    specs={name:{'family':name} for name in arms}
    for sparsity in [0.,.3,.5,.7,.8]:
        for fraction in [.25,.5,1.]:
            name=f'draft_skip{round(sparsity*100):02}_i{round(fraction*100):03}'
            arms[name]=make_draft(sparsity,fraction)
            specs[name]={'family':'draft','skip_fraction':sparsity,'retained_int8_fraction':fraction,
                         'all_video_fp4_fraction':(1-sparsity)*(1-fraction),
                         'all_video_int8_fraction':(1-sparsity)*fraction}
    arms['draft_skip70_i033']=make_draft(.7,1/3)
    specs['draft_skip70_i033']={'family':'draft','skip_fraction':.7,'retained_int8_fraction':1/3,
                              'all_video_fp4_fraction':.2,'all_video_int8_fraction':.1}
    ref=bf16()
    errors={}
    for name,fn in arms.items():
        out=fn()
        if name.startswith('draft'):
            out=out.transpose(1,2)
        assert out.shape==ref.shape and out.dtype==ref.dtype and torch.isfinite(out).all(),name
        errors[name]=metrics(out,ref,prefix)
    del out,ref
    report={'format':'e142-attention-tradeoff-v1','sample':sample,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'gpu':torch.cuda.get_device_name(),'gpu_capability':list(torch.cuda.get_device_capability()),
            'capture':str(source),'shape_hnd':list(q.shape),'case':p['case'],'step':p['step'],'layer':p['layer'],
            'video_grid':p['video_grid'],'prefix_tokens':prefix,
            'scope':'full native API including centering,quantization,draft scoring,routing,packing,attention,output assembly',
            'input_policy':'HND/BSHD contiguous layouts prebuilt; same complete QKV; no clones/transfers timed; noncausal',
            'draft_policy':'Q64,MeanMax0.2,official per-layer NVFP4 scales,INT8 prefix,0FP16 retained video blocks',
            'sampling':'25 warmups +6x12 CUDA-event calls in each forward/reverse order =144 measured calls per arm',
            'errors':errors,'specs':specs,'rounds':{name:[] for name in arms}}
    REPORT.mkdir(parents=True,exist_ok=True)
    target=REPORT/f'{sample}.json'
    assert not target.exists(),target
    for order in [list(arms),list(reversed(arms))]:
        for name in order:
            timing=measure(arms[name],lambda:None)
            report['rounds'][name].append(timing)
            target.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'sample':sample,'arm':name,'round':len(report['rounds'][name]),
                              'ms':timing['mean_ms'],'video_nmse':errors[name]['video']['nmse']}),flush=True)
    base=statistics.mean(t['mean_ms'] for t in report['rounds']['bf16_sdpa'])
    report['summary']={}
    for name,timings in report['rounds'].items():
        ms=statistics.mean(t['mean_ms'] for t in timings)
        batches=[t for r in timings for t in r['batch_means_ms']]
        report['summary'][name]={'mean_ms':ms,'speedup_over_bf16':base/ms,
                                 'cv_batch_percent':100*statistics.stdev(batches)/ms,
                                 'output_nmse':errors[name]['all']['nmse'],
                                 'video_nmse':errors[name]['video']['nmse'],**specs[name]}
    report['status']='complete'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(f'COMPLETE {sample}',flush=True)


if __name__=='__main__':
    main()
