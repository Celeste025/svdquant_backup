#!/usr/bin/env python3
"""Compare old QK-max vs experimental INT8 softmax-mean capture and reuse."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/anemoi-code'))
from anemoi import AttentionRouteCache,NVFP4Calibration,QuantConfig,SparseConfig,VisualLayout,anemoi_attention

def nmse(a,b):
 x=a.float();y=b.float();return float((x-y).double().square().sum()/y.double().square().sum())
def hashes(plan):
 import hashlib
 return {name:hashlib.sha256(getattr(plan,name).detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for name in ('block_ids','packed_offsets','nvfp4_counts','fp8_counts','fp16_counts')}
def measure(fn,repeats):
 out=fn();torch.cuda.synchronize();del out
 vals=[]
 for _ in range(repeats):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();out=fn();end.record();end.synchronize();vals.append(start.elapsed_time(end));del out
 return {'mean_ms':sum(vals)/len(vals),'samples_ms':vals}
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--include-fused',action='store_true');args=p.parse_args();assert not args.output.exists();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load(args.capture,map_location='cpu',weights_only=True,mmap=True);q,k,v=[data[n].cuda().transpose(1,2).contiguous() for n in ('q','k','v')];layer=int(data['layer']);layout=VisualLayout(tuple(data['video_grid']),int(data['prefix_tokens']))
 scales=json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales'];cal=NVFP4Calibration(*(float(scales[n][layer]) for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
 common=dict(layout=layout,layer=layer,sparse_config=SparseConfig(query_block_size=128,sparsity_ratio=0.,maxpool_weight=.2),quant_config=QuantConfig(nvfp4_ratio=.9,int8_ratio=.1,prefix_kv_precision='int8',prefix_query_precision='int8'),calibration=cal)
 modes=('qk_max','softmax_mean','softmax_mean_fused','softmax_mean_fused_k64') if args.include_fused else ('qk_max','softmax_mean')
 caches={name:AttentionRouteCache(capture_mode=name) for name in modes}
 def run(mode,first):return anemoi_attention(q,k,v,**common,first_step=first,route_cache=caches[mode])
 report={'status':'running','capture':str(args.capture),'grid':list(data['video_grid']),'prefix_tokens':data['prefix_tokens'],'layer':layer,'gpu':torch.cuda.get_device_name(),'method':'first-step dense INT8; qk_max native vs softmax_mean two-pass Triton, 10% INT8 / 90% NVFP4; reuse fixed route','arms':{}}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 first={};reuse={}
 for mode in caches:
  baseline_allocated = torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();first[mode]=run(mode,True);torch.cuda.synchronize();incremental_peak=torch.cuda.max_memory_allocated()-baseline_allocated;reuse[mode]=run(mode,False);assert torch.isfinite(first[mode]).all() and torch.isfinite(reuse[mode]).all()
  report['arms'][mode]={'cache_bytes':caches[mode].allocated_bytes,'route_hashes':hashes(caches[mode]._routes[layer].plan),'first_step_ms':measure(lambda:run(mode,True),args.repeats),'reuse_ms':measure(lambda:run(mode,False),args.repeats),'capture_incremental_peak_bytes':incremental_peak}
  print(mode,report['arms'][mode],flush=True)
 report['first_output_nmse_between_modes']=nmse(first['qk_max'],first['softmax_mean']);report['reuse_output_nmse_between_modes']=nmse(reuse['qk_max'],reuse['softmax_mean']);assert all(nmse(first['qk_max'],first[m])<1e-8 for m in modes);report['fused_first_output_nmse']=nmse(first['qk_max'],first['softmax_mean_fused']);report['fused_reuse_vs_two_pass_nmse']=nmse(reuse['softmax_mean'],reuse['softmax_mean_fused']);report['status']='complete';args.output.write_text(json.dumps(report,indent=2)+'\n');print('E200_COMPLETE',flush=True)
if __name__=='__main__':main()
