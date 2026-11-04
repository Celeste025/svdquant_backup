"""Profile native fused statistics and probability-map reduction separately."""
import argparse,json,sys,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
# Reuse the operand preparation from E203, without its validation/benchmark.
source=(ROOT/'scripts/research/validate_e203_fused_capture.py').read_text();exec(source[:source.index('a,score=')])
from anemoi.layers.attention.mpa.probability_capture_fused import extension,_reduce
ap=argparse.ArgumentParser();ap.add_argument('--ncu',action='store_true');ap.add_argument('--baseline-ncu',action='store_true');args=ap.parse_args()
q8,qs,k8,ks,v8,vs=ops;ext=extension();b,h,qn,_=q8.shape;qb=qn//128;kb=k8.size(2)//64
out,local,lse=ext.capture(*ops,valid);score=torch.empty((b,h,qb,qb),device=q8.device,dtype=torch.float16)
def native():return ext.capture(*ops,valid)
def reduction():_reduce[(qb,qb,b*h)](local,lse,lay.counts,score,qb,kb,pb,num_warps=4)
reduction();torch.cuda.synchronize()
if args.ncu or args.baseline_ncu:
 torch.cuda.cudart().cudaProfilerStart();sm120_q128_dense_int8_importance_attention(ops,valid) if args.baseline_ncu else native();torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop();raise SystemExit()
def bench(fn):
 for _ in range(3):fn()
 torch.cuda.synchronize();times=[]
 for _ in range(15):
  s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();fn();e.record();e.synchronize();times.append(s.elapsed_time(e))
 return {'median_ms':statistics.median(times),'mean_ms':statistics.mean(times),'samples_ms':times}
r={'native_fused_capture':bench(native),'probability_reduce':bench(reduction),'native_qk_max':bench(lambda:sm120_q128_dense_int8_importance_attention(ops,valid)), 'local_shape':list(local.shape),'local_bytes':local.numel()*local.element_size(),'lse_bytes':lse.numel()*lse.element_size(),'map_bytes':score.numel()*score.element_size()}
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 for _ in range(3):fused_softmax_mean(ops,valid,lay.counts,pb)
 torch.cuda.synchronize()
r['cuda_events']=[{'name':e.key,'count':e.count,'device_time_total_us':e.device_time_total} for e in prof.key_averages() if e.device_time_total>0]
path=ROOT/'results/research/E205';path.mkdir(parents=True,exist_ok=True);(path/'stages.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='cuda_events'},indent=2),flush=True)
