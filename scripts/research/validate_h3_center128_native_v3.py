#!/usr/bin/env python3
"""E082 native acceptance on actual saved H3 tensors, before any generation."""
import gc,json,os,time,traceback
from pathlib import Path
import torch
from probe_h3_plain_baseline import file_record,tree_signature
import e082_sage_center as center
from h3_sage_center_router import contract
from probe_h3_sage3_interaction import sage
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E082';DATA=Path('/data1/models/svdquant-wjq/research/20261005/E082/validation')

def guard():
 assert time.monotonic()-START<600
 assert torch.cuda.max_memory_allocated()<60*2**30

def timing(fn):
 fn();torch.cuda.synchronize();t=time.monotonic()
 for _ in range(3):fn()
 torch.cuda.synchronize();return (time.monotonic()-t)/3

@torch.inference_mode()
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 center.load_extension();report['private_contract']=contract()
 prior=json.loads((ROOT/'results/research/E081/probe.json').read_text());assert prior['status']=='complete'
 for row in prior['rows']:
  guard();source=row['source'];assert file_record(source['file'])==source
  x=torch.load(source['file'],map_location='cpu',weights_only=True,mmap=True)
  saved=torch.load(row['outputs']['file'],map_location='cpu',weights_only=True);assert file_record(row['outputs']['file'])==row['outputs']
  q,k,v=[x[t].transpose(0,1).unsqueeze(0).contiguous().cuda() for t in ('q','k','v')]
  baseline=lambda:sage.sageattn3_blackwell(q,k.clone(),v,is_causal=False,per_block_mean=True)
  zero=lambda:center.sageattn3_center128(q,k.clone(),v,x['video_start'],center=False)
  changed=lambda:center.sageattn3_center128(q,k.clone(),v,x['video_start'],center=True)
  b=baseline();z=zero();y=changed();torch.cuda.synchronize()
  captured=x['output'].transpose(0,1).unsqueeze(0).cuda()
  zero_exact=bool(torch.equal(b,z));original_exact=bool(torch.equal(b,captured))
  assert zero_exact and original_exact,(zero_exact,original_exact)
  assert y.shape==b.shape and y.dtype==b.dtype and bool(torch.isfinite(y).all())
  qi=saved['query_indices'].cuda();ys=y[0,:,qi].float().cpu();sim=saved['native_qk']['center128'];oracle=saved['native_qk']['oracle']
  gap=float((ys.double()-sim.double()).square().sum());energy=float((sim.double()-oracle.double()).square().sum());ratio=gap/max(energy,1e-30)
  tb=timing(baseline);tc=timing(changed)
  path=DATA/f'{row["case_id"]}_b{row["block"]}.pt';assert not path.exists();torch.save(dict(candidate=ys,simulator=sim,oracle=oracle,full=saved['exact_attention'],query_indices=saved['query_indices']),path)
  record=dict(case_id=row['case_id'],block=row['block'],source=source,zero_mean_byte_exact=zero_exact,original_capture_byte_exact=original_exact,simulator_gap_energy=gap,center_error_energy=energy,gap_ratio=ratio,baseline_seconds=tb,center_seconds=tc,time_ratio=tc/tb,outputs=file_record(path))
  report['rows'].append(record);save();print(json.dumps(record),flush=True)
  assert ratio<=.01,ratio;assert tc/tb<=5,(tb,tc)
  del q,k,v,b,z,y,captured,x,saved;gc.collect();torch.cuda.empty_cache()
 # Constant V checks the exact mass restoration on fully centered blocks.
 torch.manual_seed(8200);q=torch.randn(1,2,256,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.full_like(q,3.5)
 y=center.sageattn3_center128(q,k,v,0,center=True)
 report['constant_v_max_abs']=float((y.float()-3.5).abs().max());assert report['constant_v_max_abs']<.001
 report.update(status='complete',passed=True)

def save():dest.write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 START=time.monotonic();OUT.mkdir(exist_ok=True,parents=True);DATA.mkdir(exist_ok=True,parents=True);dest=OUT/'native_validation_v3.json';assert not dest.exists()
 report=dict(experiment='E082',phase='native_validation',status='running',passed=False,rows=[],sources={p.name:file_record(p) for p in (Path(__file__),ROOT/'research_state/06_experiments/E082_center128_video_plan.md')})
 try:run()
 except BaseException:report.update(status='failed_stop',error=traceback.format_exc());raise
 finally:report.update(seconds=time.monotonic()-START,peak_allocated_bytes=torch.cuda.max_memory_allocated());save()
