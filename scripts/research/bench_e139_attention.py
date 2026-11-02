#!/usr/bin/env python3
"""Paired full-API attention latency on the exact captured H3 race QKV."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
SAGE=Path('/data1/models/svdquant-wjq/research/20261025/E130/optin_rawk_readonly')
sys.path[:0]=[str(ROOT/'third_party/anemoi-code'),str(SAGE),str(ROOT/'scripts/research')]
from anemoi import anemoi_attention, SparseConfig, QuantConfig, VisualLayout, NVFP4Calibration
from sageattn3 import sageattn3_blackwell
from bench_e130_combined_lengths import CAPTURE, measure


@torch.inference_mode()
def main():
    torch.set_num_threads(6)
    p=torch.load(CAPTURE,map_location='cpu',weights_only=True,mmap=True)
    q,k,v=[p[name].cuda().contiguous() for name in ['q','k','v']]
    assert q.shape==k.shape==v.shape==(1,56,22677,128)
    qb,kb,vb=[x.transpose(1,2).contiguous() for x in [q,k,v]]
    kw=torch.empty_like(k)
    layer=25
    config=ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json'
    scales=json.loads(config.read_text())['scales']
    calibration=NVFP4Calibration(*(float(scales[n][layer]) for n in ['q_tensor_scale','k_tensor_scale','v_tensor_scale']))
    sparse=SparseConfig(query_block_size=64,sparsity_ratio=.70,maxpool_weight=.2)
    layout=VisualLayout((37,18,32),1365)
    int8=QuantConfig(nvfp4_ratio=0.,int8_ratio=1.)
    mixed=QuantConfig(nvfp4_ratio=2/3,int8_ratio=1/3)
    def bf16():
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
    def sage3():
        return sageattn3_blackwell(q,kw,v,is_causal=False,per_block_mean=True)
    def sage3_opt():
        return sageattn3_blackwell(q,k,v,is_causal=False,per_block_mean=True,
                                  fast_preprocess=True,fast_qpack=True,
                                  fast_delta_gemm=True,fast_raw_k_delta=True)
    def draft(config):
        return anemoi_attention(qb,kb,vb,layout=layout,layer=layer,
                                 sparse_config=sparse,quant_config=config,calibration=calibration)
    arms={'bf16_sdpa':(bf16,lambda:None),
          'sage3_sm110':(sage3,lambda:kw.copy_(k)),
          'sage3_optimized':(sage3_opt,lambda:None),
          'draft_s70_int8_30':(lambda:draft(int8),lambda:None),
          'draft_s70_fp4_20_int8_10':(lambda:draft(mixed),lambda:None)}
    reference=bf16()
    errors={}
    for name,(fn,prepare) in arms.items():
        prepare()
        out=fn()
        if name.startswith('draft'):
            out=out.transpose(1,2)
        assert out.shape==reference.shape and torch.isfinite(out).all()
        errors[name]=float((out.float()-reference.float()).square().sum()/reference.float().square().sum())
    del out,reference
    timings={name:[] for name in arms}
    for order in [list(arms),list(reversed(arms))]:
        for name in order:
            row=measure(*arms[name])
            timings[name].append(row)
            print(json.dumps({'arm':name,'round':len(timings[name]),**row}),flush=True)
    summary={}
    base=statistics.mean(r['mean_ms'] for r in timings['bf16_sdpa'])
    for name,rows in timings.items():
        ms=statistics.mean(r['mean_ms'] for r in rows)
        summary[name]={'mean_ms':ms,'speedup_over_bf16':base/ms,'output_nmse':errors[name],
                       'round_mean_ms':[r['mean_ms'] for r in rows],
                       'cv_batch_percent':100*statistics.stdev([x for r in rows for x in r['batch_means_ms']])/ms}
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'gpu':torch.cuda.get_device_name(),'capture':str(CAPTURE),'shape_hnd':list(q.shape),
            'video_grid':[37,18,32],'prefix_tokens':1365,'layer':layer,'sage3_source':str(SAGE),
            'scope':'full API, native dynamic preprocess + route + quantization + attention + output; no model/weight transfers',
            'layout_policy':'native contiguous HND and BSHD inputs prepared outside timing; K restore outside timing',
            'sampling':'25 warmups + 6x12 CUDA-event calls per arm per order; forward/reverse orders, 144 measured calls per arm',
            'draft_phase_scope':'sparse-phase single layer, excluding dense-first warmup steps/layers',
            'summary':summary,'rounds':timings}
    target=ROOT/'results/research/E139/attention_speed.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
