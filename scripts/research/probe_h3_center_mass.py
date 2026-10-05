#!/usr/bin/env python3
"""E081 center representation versus probability-mass compensation ablation."""
import gc,json,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import e080_pv_contract as c
from probe_h3_plain_baseline import file_record,tree_signature
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E081';OLD=ROOT/'results/research/E080/v2';DATA=Path('/data1/models/svdquant-wjq/research/20261005/E081')
SAGE=Path('/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004/sageattention3_blackwell')
import sys
sys.path.insert(0,str(SAGE))
import sageattn3.api as sage
ARMS=('base','center32','center128','residual32','residual128','massfix32','massfix128')

def stats(y,ref):
 y=y.double();ref=ref.double();e=y-ref
 d=y[:,1::2]-y[:,0::2];rd=ref[:,1::2]-ref[:,0::2];de=d-rd
 return dict(sse=float(e.square().sum()),reference_energy=float(ref.square().sum()),contrast_sse=float(de.square().sum()),contrast_reference_energy=float(rd.square().sum()),contrast_gain=float((d*rd).sum()/rd.square().sum()),contrast_cosine=float((d*rd).sum()/(d.norm()*rd.norm()).clamp_min(1e-30)))

def verify_pack(values,pack,kind):
 native=c.decode_packed(*pack,kind)
 target=values.transpose(-2,-1).contiguous() if kind=='v' else values
 qdq,details=c.v_quant_dequant(target,return_details=True)
 if kind=='k':
  perm=c.k_permutation(values.shape[-2],values.device)
  codes=details['codes'].index_select(-2,perm)
  scales=details['scales'].index_select(-2,perm)
 else:codes=details['codes'];scales=details['scales']
 expected_codes=torch.stack((pack[0]&15,pack[0]>>4),-1).flatten(-2)
 byte_errors=int((codes!=expected_codes).sum());scale_errors=int((scales.view(torch.uint8)!=c.logical_scales(pack[1]).view(torch.uint8)).sum())
 if kind=='v':qdq=qdq.transpose(-2,-1)
 dq_errors=int((qdq!=native).sum())
 assert byte_errors==scale_errors==dq_errors==0,(kind,byte_errors,scale_errors,dq_errors)
 return native,dict(code_mismatches=byte_errors,scale_mismatches=scale_errors,dq_mismatches=dq_errors)

def transform(x,mask,side):
 return torch.where(mask,c.pair_forward(x,side=side),x)

def inverse(x,mask):return torch.where(mask,c.inverse_P(x),x)

def vq(v):return c.v_quant_dequant(v.transpose(-2,-1).contiguous()).transpose(-2,-1).contiguous()

@torch.inference_mode()
def run_case(row):
 guard();source=file_record(row['artifact']['file']);assert source==row['artifact']
 x=torch.load(source['file'],map_location='cpu',weights_only=True,mmap=True)
 assert tree_signature(x)==row['tensors']
 n=x['valid_length'];vs=x['video_start'];N=(n+127)//128*128;H=56;D=128
 assert x['video_grid']==[37,18,32] and n-vs==21312
 assert all(tuple(x[t].shape)==(n,H,D) for t in ('q','k','v','output'))
 qi=torch.tensor([vs+(f*18+h)*32+w+dx for f in (0,12,24,36) for h in (4,9,13) for w in (7,15,23) for dx in (0,1)],device='cuda')
 assert len(qi)==72 and int(qi.max())<n
 q,k,v=[x[name].transpose(0,1).unsqueeze(0).contiguous().cuda() for name in ('q','k','v')]
 native=x['output'].transpose(0,1).cuda()[:,qi].float()
 exact_logits=(q[0,:,qi].float()@k[0].float().transpose(-2,-1))*(D**-.5)
 exact_o=(torch.softmax(exact_logits.double(),-1)@v[0].double()).float()
 del exact_logits
 qp,kp,vp,ds=sage.preprocess_qkv(q.clone(),k.clone(),v.clone(),per_block_mean=True)
 packs=[sage.scale_and_quant_fp4(qp),sage.scale_and_quant_fp4_permute(kp),sage.scale_and_quant_fp4_transpose(vp)]
 deq=[];checks={}
 for name,values,pack in zip(('q','k','v'),(qp,kp,vp),packs):
  z,check=verify_pack(values,pack,name);deq.append(z[0]);checks[name]=check
 qd,kd,vd=deq
 del packs,qp,kp,ds
 # delta_s uses BF16 matmul in the official preprocess. Recreate it once.
 qp,kp,vp,ds=sage.preprocess_qkv(q.clone(),k.clone(),v.clone(),per_block_mean=True)
 native_logits=(qd[:,qi]@kd.transpose(-2,-1)+ds[0,:,qi//128])*(D**-.5)
 native_logits[...,n:]=-torch.inf
 del qp,kp,ds,qd,kd,deq
 vf=vp[0].float();del vp
 values={'base':vd}; means={}
 for g in (32,128):
  gmask=((torch.arange(N//g,device='cuda')*g>=vs)&((torch.arange(N//g,device='cuda')+1)*g<=n))
  mu=vf.reshape(H,N//g,g,D).mean(-2).to(torch.bfloat16).float()*gmask[None,:,None]
  residual=vf-mu.repeat_interleave(g,dim=1)
  values['center'+str(g)]=vq(residual);values['residual'+str(g)]=values['center'+str(g)];means[g]=mu
 record=dict(case_id=x['case_id'],block=x['block'],step=x['step'],source=source,heads=list(range(56)),query_indices=qi.cpu().tolist(),pack_parity=checks,conditions=[])
 old_report=json.loads((OLD/'pv_probe_v2.json').read_text())
 old_row=next(r for r in old_report['rows'] if r['case_id']==x['case_id'] and r['block']==x['block'])
 assert file_record(old_row['outputs']['file'])==old_row['outputs']
 old=torch.load(old_row['outputs']['file'],map_location='cpu',weights_only=True)
 saved=dict(exact_attention=exact_o.cpu(),native_capture=native.cpu(),query_indices=qi.cpu())
 for condition in ('native_qk','exact_qk'):
  guard()
  if condition=='native_qk':logits=native_logits
  else:
   logits=(q[0,:,qi].float()@k[0].float().transpose(-2,-1))*(D**-.5)
   logits=F.pad(logits,(0,N-n),value=-torch.inf)
  oracle64=torch.softmax(logits.double(),-1)@vf.double();oracle=oracle64.float()
  accum={arm:torch.zeros((H,len(qi),D),device='cuda') for arm in ARMS}
  corrections={g:torch.zeros_like(oracle) for g in (32,128)}
  acc64=torch.zeros_like(oracle64);den64=torch.zeros((H,len(qi)),device='cuda',dtype=torch.float64);max64=torch.full_like(den64,-torch.inf)
  acc_exact=torch.zeros_like(oracle);den=torch.zeros((H,len(qi)),device='cuda');maximum=torch.full_like(den,-torch.inf)
  for start in range(N-128,-1,-128):
   guard();end=start+128;score=logits[:,:,start:end];newmax=torch.maximum(maximum,score.amax(-1));rescale=torch.exp(maximum-newmax)
   u=torch.exp(score-newmax[...,None])*2688.;den=den*rescale+u.sum(-1);maximum=newmax
   sl=slice(start,end);up=c.p_quant_dequant(u)
   zb=up@values['base'][:,sl,:]
   accum['base']=accum['base']*rescale[...,None]+zb
   for g in (32,128):
    mass=u.reshape(H,len(qi),128//g,g).sum(-1)
    qmass=up.reshape(H,len(qi),128//g,g).sum(-1)
    mu=means[g][:,start//g:end//g,:]
    zr=up@values['center'+str(g)][:,sl,:]
    cm=(mass-qmass)@mu
    zc=zr+mass@mu;zq=zr+qmass@mu;zm=zb+cm
    for arm,z in [('center'+str(g),zc),('residual'+str(g),zq),('massfix'+str(g),zm)]:
     accum[arm]=accum[arm]*rescale[...,None]+z
    corrections[g]=corrections[g]*rescale[...,None]+cm
   acc_exact=acc_exact*rescale[...,None]+u@vf[:,sl,:]
   score64=score.double();next64=torch.maximum(max64,score64.amax(-1));rescale64=torch.exp(max64-next64)
   u64=torch.exp(score64-next64[...,None])*2688.;den64=den64*rescale64+u64.sum(-1);max64=next64
   acc64=acc64*rescale64[...,None]+u64@vf[:,sl,:].double()
  high=acc_exact/den[...,None]
  online_gap=(high-oracle).abs().max().item()
  double_gap=(acc64/den64[...,None]-oracle64).abs().max().item();assert double_gap<1e-8,double_gap
  result=dict(condition=condition,oracle_vs_full=stats(oracle,exact_o),online_max_abs=online_gap,fp64_online_max_abs=double_gap,arms={})
  saved[condition]={'oracle':oracle.cpu()}
  assert torch.equal(oracle.cpu(),old[condition]['oracle'])
  for arm in ARMS:
   y=accum[arm]/den[...,None];saved[condition][arm]=y.cpu()
   result['arms'][arm]=dict(vs_pv_oracle=stats(y,oracle),vs_full_attention=stats(y,exact_o))
  replayed={a:bool(torch.equal(saved[condition][a],old[condition][a])) for a in ('base','center32','center128')}
  assert all(replayed.values()),replayed
  result['E080_replay_exact']=replayed
  result['correction_identity_relative_energy']={}
  result['factorial_closure_relative_energy']={}
  for g in (32,128):
   corr=(corrections[g]/den[...,None]).cpu();saved[condition]['correction'+str(g)]=corr
   delta=saved[condition]['center'+str(g)]-saved[condition]['residual'+str(g)]
   gap=float((delta.double()-corr.double()).square().sum()/corr.double().square().sum().clamp_min(1e-30))
   assert gap<1e-8,gap
   result['correction_identity_relative_energy'][str(g)]=gap
   b=saved[condition]['base'].double();dv=saved[condition]['residual'+str(g)].double()-b;dm=saved[condition]['massfix'+str(g)].double()-b;ce=saved[condition]['center'+str(g)].double()
   closure=float((ce-b-dv-dm).square().sum()/(ce-b).square().sum().clamp_min(1e-30));assert closure<1e-8,closure
   result['factorial_closure_relative_energy'][str(g)]=closure
  sim=saved[condition]['base'].cuda().to(torch.bfloat16).float()
  result['base_simulator_vs_native']=stats(sim,native)
  result['native_vs_full']=stats(native,exact_o)
  result['native_gap_energy_ratio']=result['base_simulator_vs_native']['sse']/max(result['native_vs_full']['sse'],1e-30)
  record['conditions'].append(result)
  print(json.dumps(dict(case=x['case_id'],block=x['block'],condition=condition,gap_ratio=result['native_gap_energy_ratio'],contrast={a:result['arms'][a]['vs_pv_oracle']['contrast_sse'] for a in ARMS})),flush=True)
  del logits,oracle,oracle64,accum,acc_exact,acc64,den64,max64,corrections
 path=DATA/f'center_{x["case_id"]}_b{x["block"]}.pt';assert not path.exists();torch.save(saved,path);record['outputs']=file_record(path)
 return record

def guard():
 assert time.monotonic()-START<1200,'Local analysis budget exhausted'
 assert torch.cuda.max_memory_allocated()<60*2**30,'Memory budget exceeded'

if __name__=='__main__':
 START=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);dest=OUT/'probe.json';assert not dest.exists()
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 report=dict(experiment='E081',phase='center_mass_ablation',status='running',sources={p.name:file_record(p) for p in [Path(__file__),ROOT/'scripts/research/e080_pv_contract.py',ROOT/'research_state/06_experiments/E081_center_mass_plan.md']},rows=[])
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
  for r in (0,1):
   capture=json.loads((OLD/f'capture_r{r}.json').read_text());assert capture['status']=='complete' and capture['final_replay_exact']
   for row in capture['captures']:
    report['rows'].append(run_case(row));dest.write_text(json.dumps(report,indent=2));gc.collect();torch.cuda.empty_cache()
  report['status']='complete'
 except BaseException:report.update(status='failed_stop',error=traceback.format_exc());raise
 finally:
  report.update(seconds=time.monotonic()-START,peak_allocated_bytes=torch.cuda.max_memory_allocated());dest.write_text(json.dumps(report,indent=2))
