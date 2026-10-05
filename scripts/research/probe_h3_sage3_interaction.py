#!/usr/bin/env python3
"""E078 official Sage3 / frozen SVDQuant four-corner full-H3 probe."""
import gc,importlib,json,sys,time,traceback,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
SAGE=Path('/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004/sageattention3_blackwell')
sys.path.insert(0,str(HERE));sys.path.insert(0,str(SAGE))
import torch
import run_h3_native_paired_video as e010
import probe_h3_plain_baseline as base
import probe_h3_lowrank_initialization as refs
from h3_native_nvfp4 import install_native_h3
from h3_native_fp4_attention import H3AttentionRouter
import sageattn3.api as sage
OUT=ROOT/'results/research/E078';DATA=Path('/data1/models/svdquant-wjq/research/20261004/E078')
PLAN=ROOT/'research_state/06_experiments/E078_sage3_interaction_plan.md'

def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 if isinstance(x,list):return [move(v) for v in x]
 if isinstance(x,tuple):return tuple(move(v) for v in x)
 return x

class Router(H3AttentionRouter):
 def __init__(self,dit):
  super().__init__(dit,'block_mean',_official=sage)
  self.native_calls=0
 def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
  log=self._active_log.get();block=self._active_main.get()
  assert log is not None and softmax_scale==128**-.5
  expected=log.expected_refiner_cu if block is None else log.expected_cu
  log.remember_cu(cu_seqlens,expected,'refiner' if block is None else 'main')
  if block is None:
   log.refiner_helpers+=1;log.original_bf16_segments+=1
   return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
  n,total=expected[1:];assert q.shape==k.shape==v.shape==(total,56,128)
  assert all(x.dtype==torch.bfloat16 for x in (q,k,v))
  Q,K,V=[x[:n].transpose(0,1).unsqueeze(0).contiguous() for x in (q,k,v)]
  # Official preprocessing modifies K in place: own storage even if contiguous.
  K=K.clone()
  result=sage.sageattn3_blackwell(Q,K,V,is_causal=False,per_block_mean=True)
  assert result.shape==Q.shape and result.dtype==q.dtype
  out=torch.empty_like(q);out[:n]=result[0].transpose(0,1)
  out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
  log.main_rows.append(dict(block=block,valid_length=n,padding_length=total-n))
  log.fp4_calls+=1;log.original_bf16_segments+=1;log.finite(out)
  return out

def save(): (OUT/'run.json').write_text(json.dumps(report,indent=2)+'\n')
def guard():
 assert time.time()-start<600
 assert report['calls']<=8 and torch.cuda.max_memory_allocated()<60*1024**3

@torch.inference_mode()
def run():
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 original=sage.fp4attn_cuda.fwd
 counter=[0]
 def counted(*args,**kw):
  counter[0]+=1;return original(*args,**kw)
 sage.fp4attn_cuda.fwd=counted
 parent_path=ROOT/'results/research/E065b/evaluate.json';parent=json.loads(parent_path.read_text())
 report['parent']=base.file_record(parent_path)
 report['sage_commit']=subprocess.check_output(['git','-C',str(SAGE),'rev-parse','HEAD'],text=True).strip()
 report['cutlass_dependency']=dict(path=str((SAGE/'csrc/cutlass').resolve()),version_header=base.file_record(SAGE/'csrc/cutlass/include/cutlass/version.h'),version='4.5.0 cached FlashInfer dependency; no git metadata')
 report['sage_api']=base.file_record(sage.__file__)
 report['extensions']=[base.file_record(x.__file__) for x in (sage.fp4attn_cuda,sage.fp4quant_cuda)]
 cases=[c for c in parent['cases'] if c['arm']=='restart' and c['position']=='source_teacher'];assert len(cases)==2
 sources={};reference={}
 for c in cases:
  key=c['case_id']+'/'+c['position'];sources[key]=torch.load(base.verify_file(c['artifact']),map_location='cpu',weights_only=True,mmap=True)
  assert base.tree_signature(sources[key]['actual_dit_inputs'])==c['actual_dit_input_signature']
  reference[key]={a:refs.load_reference(parent['references'][key][a],c['actual_dit_input_signature'])['raw_outputs'] for a in ('bf16','legacy_selected')}
 # Tiny nonaligned smoke checks real extension and unchanged caller K storage.
 torch.manual_seed(778)
 x=[torch.randn(1,2,257,128,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
 result=sage.sageattn3_blackwell(x[0],x[1].clone(),x[2],is_causal=False)
 assert result.shape==x[0].shape and bool(torch.isfinite(result).all()) and counter[0]==1
 report['smoke']=dict(shape=list(result.shape),finite=True,extension_calls=counter[0]);del x,result
 pipe=e010.load_h3_pipeline(full=False,vram_limit_gib=30.);pipe.load_models_to_device(['dit']);pipe.dit.eval()
 report['resident']=e010.make_h3_resident(pipe.dit)
 outputs={}
 for projection in ('B','S'):
  if projection=='S':
   before=base.non_target_identity(pipe.dit)
   path=base.verify_file(parent['legacy_manifest']).parent
   report['installation']=install_native_h3(pipe.dit,path,activation_packer=base.pack_activation_fast,chunk_rows=1024)
   assert before==base.non_target_identity(pipe.dit)
   gc.collect();torch.cuda.empty_cache()
  for attn in ('0','A'):
   arm=projection+attn;outputs[arm]={}
   router=Router(pipe.dit) if attn=='A' else None
   for c in cases:
    guard();key=c['case_id']+'/'+c['position'];tree=move(sources[key]['actual_dit_inputs']);kw=tree['kwargs']
    contract=dict(expected_cu=kw['packed_seq_params']['cu_seqlens_q'].cpu().tolist(),expected_refiner_cu=kw['refiner_packed_seq_params']['cu_seqlens_q'].cpu().tolist())
    from contextlib import nullcontext
    audit=base.RuntimeAudit();audit.phase='native';n0=counter[0]
    report['attempted_calls']+=1;save()
    with audit.installed(), (router.forward_context(**contract,diagnostics=True) if router else nullcontext()) as alog:
     result=pipe.dit(*tree['args'],**kw);torch.cuda.synchronize()
    report['calls']+=1
    raw,signatures=base.cpu_outputs(result);runtime=audit.row()
    assert runtime['sdpa_calls']==(52 if router else 102)
    assert runtime['scaled_mm_calls']==(200 if projection=='S' else 0) and runtime['disk_loads']==0
    assert counter[0]-n0==(50 if router else 0)
    replay=None
    if attn=='0':
     ref=reference[key]['bf16' if projection=='B' else 'legacy_selected']
     replay=all(torch.equal(raw[m],ref[m]) for m in raw);assert replay,'baseline replay mismatch'
    outputs[arm][key]=raw
    path=DATA/(arm+'_'+key.replace('/','_')+'.pt');torch.save(raw,path)
    metric={m:float((v.double()-reference[key]['bf16'][m].double()).square().sum()) for m,v in raw.items()}
    report['cases'].append(dict(arm=arm,key=key,raw_sse=metric,replay_exact=replay,artifact=base.file_record(path),runtime=runtime,sage_extension_calls=counter[0]-n0,attention=None if not alog else alog.summary))
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save();print(arm,key,metric,flush=True)
    del result,tree,kw
   if router:router.close()
 report['comparisons']=[]
 for key in sources:
  row={'key':key}
  for m in ('video','audio'):
   b=outputs['B0'][key][m].double();s=outputs['S0'][key][m].double()-b;a=outputs['BA'][key][m].double()-b;both=outputs['SA'][key][m].double()-b
   interaction=both-s-a;energy=lambda x:float(x.square().sum())
   row[m]=dict(svd_sse=energy(s),sage_sse=energy(a),combined_sse=energy(both),additive_sse=energy(s+a),net_amplification=energy(both)/energy(s+a),combined_vs_svd=energy(both)/energy(s),attention_increment_ratio=energy(both-s)/energy(a),interaction_sse=energy(interaction),cos_single_errors=float((s*a).sum())/(energy(s)*energy(a))**.5)
  report['comparisons'].append(row)
 assert report['calls']==report['attempted_calls']==8
 report.update(status='complete',seconds=time.time()-start,total_sage_extension_calls=counter[0]);save()

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'run.json').exists();DATA.mkdir(parents=True,exist_ok=False)
 start=time.time();report=dict(status='running',calls=0,attempted_calls=0,cases=[],driver=base.file_record(__file__),plan=base.file_record(PLAN))
 try:run()
 except BaseException:
  report.update(status='failed',error=traceback.format_exc(),seconds=time.time()-start);save();raise
