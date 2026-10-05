#!/usr/bin/env python3
"""E080 bounded real-QKV PV representation experiment; not a fast kernel."""
import gc,json,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import e080_pv_contract as c
from probe_h3_plain_baseline import file_record,tree_signature
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E080/v2';DATA=Path('/data1/models/svdquant-wjq/research/20261004/E080/v2')
SAGE=Path('/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004/sageattention3_blackwell')
import sys
sys.path.insert(0,str(SAGE))
import sageattn3.api as sage
ARMS=('base','center32','center128','p_only','pair','random_pair','shuffle')

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
 qi=torch.tensor([vs+(f*18+h)*32+w+dx for f in (0,12,24,36) for h in (4,9,13) for w in (7,15,23) for dx in (0,1)],device='cuda')
 assert len(qi)==72 and int(qi.max())<n
 q,k,v=[x[name].transpose(0,1).unsqueeze(0).contiguous().cuda() for name in ('q','k','v')]
 native=x['output'].transpose(0,1).cuda()[:,qi].float()
 exact_logits=(q[0,:,qi].float()@k[0].float().transpose(-2,-1))*(D**-.5)
 exact_o=torch.softmax(exact_logits,-1)@v[0].float()
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
 groups=torch.arange(N//32,device='cuda');eligible=(groups*32>=vs)&((groups+1)*32<=n)
 mask=eligible.repeat_interleave(32)
 perm=c.within32_permutation(N,'cuda');perm=torch.where(mask,perm,torch.arange(N,device='cuda'))
 vt=transform(vf.transpose(-2,-1),mask,'v').transpose(-2,-1).contiguous()
 vr=vf[:,perm,:];vrt=transform(vr.transpose(-2,-1),mask,'v').transpose(-2,-1).contiguous()
 values={'base':vd,'p_only':vd,'pair':vq(vt),'random_pair':vq(vrt),'shuffle':vq(vr)}
 means={}
 for g in (32,128):
  gmask=((torch.arange(N//g,device='cuda')*g>=vs)&((torch.arange(N//g,device='cuda')+1)*g<=n))
  mu=vf.reshape(H,N//g,g,D).mean(-2).to(torch.bfloat16).float()*gmask[None,:,None]
  residual=vf-mu.repeat_interleave(g,dim=1)
  values['center'+str(g)]=vq(residual);means[g]=mu
 # Exact algebra check uses actual first tile in FP32, distinct from quantization.
 ts=((vs+127)//128)*128
 tp=torch.softmax(native_logits[:,:,ts:ts+128],-1)
 tm=mask[ts:ts+128]
 algebra=(tp@vf[:,ts:ts+128]-transform(tp,tm,'p')@vt[:,ts:ts+128]).abs().max().item()
 assert algebra<2e-4,algebra
 # Mapping coverage: physical adjacency occasionally wraps a raster row.
 ap=(groups[eligible]*32)[:,None]+torch.arange(0,32,2,device='cuda')[None,:]
 wraps=int((((ap-vs)//32)!=((ap+1-vs)//32)).sum());total_pairs=ap.numel()
 record=dict(case_id=x['case_id'],block=x['block'],step=x['step'],source=source,heads=list(range(56)),query_indices=qi.cpu().tolist(),eligible_pairs=total_pairs,raster_row_crossing_pairs=wraps,pack_parity=checks,fp32_pair_identity_max_abs=algebra,conditions=[])
 saved=dict(exact_attention=exact_o.cpu(),native_capture=native.cpu(),query_indices=qi.cpu())
 for condition in ('native_qk','exact_qk'):
  guard()
  if condition=='native_qk':logits=native_logits
  else:
   logits=(q[0,:,qi].float()@k[0].float().transpose(-2,-1))*(D**-.5)
   logits=F.pad(logits,(0,N-n),value=-torch.inf)
  oracle=torch.softmax(logits,-1)@vf
  accum={arm:torch.zeros((H,len(qi),D),device='cuda') for arm in ARMS}
  acc_exact=torch.zeros_like(oracle);den=torch.zeros((H,len(qi)),device='cuda');maximum=torch.full_like(den,-torch.inf)
  for start in range(N-128,-1,-128):
   guard();end=start+128;score=logits[:,:,start:end];newmax=torch.maximum(maximum,score.amax(-1));rescale=torch.exp(maximum-newmax)
   u=torch.exp(score-newmax[...,None])*2688.;den=den*rescale+u.sum(-1);maximum=newmax
   sl=slice(start,end);ms=mask[sl];up=c.p_quant_dequant(u);ut=transform(u,ms,'p');utq=c.p_quant_dequant(ut)
   local_perm=perm[sl]-start;ur=u[:,:,local_perm]
   uv={'base':up,'center32':up,'center128':up,'p_only':inverse(utq,ms),'pair':utq,'random_pair':c.p_quant_dequant(transform(ur,ms,'p')),'shuffle':c.p_quant_dequant(ur)}
   for arm in ARMS:
    z=uv[arm]@values[arm][:,sl,:]
    if arm.startswith('center'):
     g=int(arm[6:]);mass=u.reshape(H,len(qi),128//g,g).sum(-1)
     z=z+mass@means[g][:,start//g:end//g,:]
    accum[arm]=accum[arm]*rescale[...,None]+z
   acc_exact=acc_exact*rescale[...,None]+u@vf[:,sl,:]
  high=acc_exact/den[...,None]
  online_gap=(high-oracle).abs().max().item();assert online_gap<3e-4,online_gap
  result=dict(condition=condition,oracle_vs_full=stats(oracle,exact_o),online_max_abs=online_gap,arms={})
  saved[condition]={'oracle':oracle.cpu()}
  for arm in ARMS:
   y=accum[arm]/den[...,None];saved[condition][arm]=y.cpu()
   result['arms'][arm]=dict(vs_pv_oracle=stats(y,oracle),vs_full_attention=stats(y,exact_o))
  sim=saved[condition]['base'].cuda().to(torch.bfloat16).float()
  result['base_simulator_vs_native']=stats(sim,native)
  result['native_vs_full']=stats(native,exact_o)
  result['native_gap_energy_ratio']=result['base_simulator_vs_native']['sse']/max(result['native_vs_full']['sse'],1e-30)
  record['conditions'].append(result)
  print(json.dumps(dict(case=x['case_id'],block=x['block'],condition=condition,gap_ratio=result['native_gap_energy_ratio'],contrast={a:result['arms'][a]['vs_pv_oracle']['contrast_sse'] for a in ARMS})),flush=True)
  del logits,oracle,accum,acc_exact,uv
 path=DATA/f'pv_{x["case_id"]}_b{x["block"]}.pt';assert not path.exists();torch.save(saved,path);record['outputs']=file_record(path)
 return record

def guard():
 assert time.monotonic()-START<1200,'Local analysis budget exhausted'
 assert torch.cuda.max_memory_allocated()<60*2**30,'Memory budget exceeded'

if __name__=='__main__':
 START=time.monotonic();dest=OUT/'pv_probe.json';assert not dest.exists()
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 report=dict(experiment='E080',phase='local_pv',status='running',sources={p.name:file_record(p) for p in [Path(__file__),ROOT/'scripts/research/e080_pv_contract.py',ROOT/'research_state/06_experiments/E080_pv_contrast_plan.md']},rows=[])
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
  for r in (0,1):
   capture=json.loads((OUT/f'capture_r{r}.json').read_text());assert capture['status']=='complete' and capture['final_replay_exact']
   for row in capture['captures']:
    report['rows'].append(run_case(row));dest.write_text(json.dumps(report,indent=2));gc.collect();torch.cuda.empty_cache()
  report['status']='complete'
 except BaseException:report.update(status='failed_stop',error=traceback.format_exc());raise
 finally:
  report.update(seconds=time.monotonic()-START,peak_allocated_bytes=torch.cuda.max_memory_allocated());dest.write_text(json.dumps(report,indent=2))
