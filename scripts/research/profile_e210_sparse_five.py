"""Matched Q128/K128, Q128/K64 and Q64/K64 operator measurements."""
import argparse,json,sys,statistics,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from anemoi import AttentionRouteCache,SparseConfig,QuantConfig,NVFP4Calibration,VisualLayout,anemoi_attention
from anemoi.layers.attention.mpa.executor import _prepare_h3_sm120_inputs
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.probability_capture_fused import fused_softmax_mean_k64

def nmse(x,y):return float((x.float()-y.float()).square().sum()/y.float().square().sum())
def bench(fn):
 for _ in range(2):fn()
 torch.cuda.synchronize();v=[]
 for _ in range(8):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();v.append(a.elapsed_time(b))
 return {'mean_ms':statistics.mean(v),'samples_ms':v}
@torch.inference_mode()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--ncu',action='store_true');args=parser.parse_args()
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 path=ROOT/'results/research/E210/result.json'
 if not args.ncu:assert not path.exists()
 data=torch.load('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory/race_480p_124f_step0_block25.pt',map_location='cpu',weights_only=True,mmap=True)
 hq,hk,hv=[data[n].cuda().contiguous() for n in ('q','k','v')];q,k,v=[x.transpose(1,2).contiguous() for x in (hq,hk,hv)];prefix=int(data['prefix_tokens']);grid=tuple(data['video_grid']);layer=int(data['layer'])
 scales=json.loads((ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales'];cal=NVFP4Calibration(*(float(scales[n][layer]) for n in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))

 report={'status':'running','input':'E187 step0 layer25 480p 124 frames full56 heads','budget':'80% skipped /10% NVFP4/10% INT8; first capture remains dense INT8','arms':{}}
 for label,qb,kb,mode in (('prob_Q128K128',128,128,'softmax_mean_fused'),('prob_Q128K64',128,64,'softmax_mean_fused_k64'),('prob_Q64K64',64,64,'softmax_mean_fused_k64'),('original_Q128K128',128,128,None),('original_Q64K64',64,64,None)):
  cache=AttentionRouteCache(capture_mode=mode) if mode else None
  kwargs=dict(layout=VisualLayout(grid,prefix),layer=layer,sparse_config=SparseConfig(query_block_size=qb,sparsity_ratio=.8,maxpool_weight=.2),quant_config=QuantConfig(nvfp4_ratio=.5,int8_ratio=.5,prefix_query_precision='int8',prefix_kv_precision='int8'),calibration=cal)
  def first():return anemoi_attention(q,k,v,**kwargs,**(dict(first_step=True,route_cache=cache) if cache is not None else {}))
  def later():return anemoi_attention(q,k,v,**kwargs,**(dict(first_step=False,route_cache=cache) if cache is not None else {}))
  a=first();b=later();assert torch.isfinite(a).all() and torch.isfinite(b).all();del a,b
  if args.ncu:
   later();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart();later();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();continue
  row={'first':bench(first),'later':bench(later),'cache_bytes':cache.allocated_bytes if cache else 0}
  if cache:
   plan=cache._routes[layer].plan;row['int8_tiles_per_head']=plan.fp8_blocks_per_head;row['nvfp4_tiles_per_head']=plan.nvfp4_blocks_per_head;row['map_shape']=[plan.fp8_counts.shape[-1],plan.logical_columns]
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   later();torch.cuda.synchronize()
  row['later_kernel_profile']=[{'name':e.key,'calls':e.count,'cuda_total_us':e.device_time_total} for e in prof.key_averages() if e.device_time_total>0]
  report['arms'][label]=row;path.write_text(json.dumps(report,indent=2));print(label,row['first']['mean_ms'],row['later']['mean_ms'],row['cache_bytes'],flush=True)
 if not args.ncu:report['status']='complete';path.write_text(json.dumps(report,indent=2));print('E210_COMPLETE',flush=True)
if __name__=='__main__':main()
