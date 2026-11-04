#!/usr/bin/env python3
"""Measure old-route mixed output with simultaneous new-route capture on real QKV."""
import json,sys,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from anemoi import AttentionRouteCache,SparseConfig,QuantConfig,NVFP4Calibration,VisualLayout,anemoi_attention

def nmse(a,b):return float((a.float()-b.float()).square().sum()/b.float().square().sum())
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
 q,k,v=[data[n].cuda().transpose(1,2).contiguous() for n in ('q','k','v')];layer=int(data['layer']);grid=tuple(data['video_grid']);prefix=int(data['prefix_tokens'])
 scales=json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales'];cal=NVFP4Calibration(*(float(scales[n][layer]) for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
 args=dict(layout=VisualLayout(grid,prefix),layer=layer,sparse_config=SparseConfig(query_block_size=64,sparsity_ratio=0.,maxpool_weight=.2),quant_config=QuantConfig(nvfp4_ratio=.9,int8_ratio=.1,prefix_query_precision='int8',prefix_kv_precision='int8'),calibration=cal)
 old=AttentionRouteCache(capture_mode='softmax_mean_fused_k64');new=AttentionRouteCache(capture_mode='softmax_mean_fused_k64')
 def capture(cache):return anemoi_attention(q,k,v,**args,first_step=True,route_cache=cache)
 def mixed():return anemoi_attention(q,k,v,**args,first_step=False,route_cache=old)
 capture(old);capture(new);ref=mixed();stream=torch.cuda.Stream()
 def overlap():
  default=torch.cuda.current_stream();stream.wait_stream(default)
  with torch.cuda.stream(stream): tmp=capture(new)
  out=mixed();default.wait_stream(stream)
  return out,tmp
 out,tmp=overlap();torch.cuda.synchronize();error=nmse(out,ref);assert error<1e-12,error
 # Same QKV in both routes: compare compact route tensors and counts exactly.
 a=old._routes[layer].plan;b=new._routes[layer].plan
 fields={}
 for name in ('block_ids','fp8_counts','nvfp4_counts','fp16_counts'):
  if hasattr(a,name):fields[name]=bool(torch.equal(getattr(a,name),getattr(b,name)))
 assert all(fields.values()),fields
 def measure(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();samples=[]
  for _ in range(8):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
  return {'mean_ms':statistics.mean(samples),'samples_ms':samples}
 with torch.inference_mode():
  result={'status':'complete','input':'E187 race 480x832 124f step0 layer25 Q64, all 56 heads','baseline':'old route mixed 10% INT8 90% NVFP4, no skip','candidate':'separate full dense INT8 probability capture overlapped with old-route mixed output; outputs joined','mixed_output_nmse':error,'route_fields_equal':fields}
  result['mixed']=measure(mixed);result['capture']=measure(lambda:capture(new));result['overlap']=measure(overlap);result['overhead_percent']=100*(result['overlap']['mean_ms']/result['mixed']['mean_ms']-1)
 path=ROOT/'results/research/E220/overlap.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
