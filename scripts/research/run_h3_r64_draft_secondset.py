#!/usr/bin/env python3
"""Three H3 precision schedules, native NVFP4 and Anemoi, matched to E150."""
import argparse, contextlib, importlib, json, math, os, sys, types
from itertools import zip_longest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
sys.path.extend(['/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/lib/python3.12/site-packages','/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13','/data1/models/svdquant-wjq/research/deps/cutlass-dsl-4.6.2-cu13/nvidia_cutlass_dsl/dsl_packages'])
import run_h3_r64_secondset_base as base
import run_e139_draftattention2 as old
from run_e159_route_reuse_whitebox import Router as LayoutRouter
from bench_h3_native_nvfp4 import make_h3_resident
from h3_native_nvfp4_r64 import install_native_h3, NativeH3Linear, TARGET_NAMES
from h3_plain_native_nvfp4 import install_plain_h3, PlainNativeH3Linear
from h3_nvfp4_zero_sf_compat import pack_activation_fast, collect_fastpack_checks
from h3_native_fp4_attention import AttentionCallLog
from anemoi import SparseConfig, QuantConfig, VisualLayout, anemoi_attention
from flashinfer.gemm import mm_nvfp4_svdquant
from h3_w8a8_group128 import W8A8Linear
import e122_h3_helpers as sage
EXPORT=Path('/data1/models/svdquant-wjq/research/20261102/h3_svdquant_rank64_native_export')
PLAIN=Path('/data1/models/svdquant-wjq/research/20261024/E128_plain_export_rehydrated')
OUT=ROOT/'results/research/h3_r64_draft_secondset'
DATA=Path('/data1/models/svdquant-wjq/research/20261102/h3_r64_draft_secondset')
STATE={'step':-1,'variant':None,'calls':0,'probe':False,'modules':[],'precision':'quant'}
class ProbeComplete(Exception):pass

def fused_forward(self,x):
 if STATE['step'] in ({0,19} if STATE['variant']=='c2' else {0,1,18,19}):
  return F.linear(x,self.bf16_weight,self.bias)
 STATE['calls']+=1
 x_s=x/self.smooth
 packet=self.pack_input(x_s,packer=pack_activation_fast)
 down=F.linear(x_s,self.lr_a).reshape(-1,64).contiguous()
 alpha=(packet.global_scale*self.weight_global).reshape(1)
 l1=(self.lr_b.float()/alpha).to(torch.bfloat16).contiguous()
 y=mm_nvfp4_svdquant(packet.packed,self.weight_packed,packet.swizzled_scales.view(torch.uint8).reshape(-1),self.weight_scales_swizzled.view(torch.uint8).reshape(-1),alpha,down,l1,bias=self.bias,backend='cute-dsl',enable_pdl=False)
 return y.reshape(*x.shape[:-1],self.out_features)

def plain_forward(self,x):
 if STATE['step'] in {0,1,18,19}:return F.linear(x,self.bf16_weight,self.bias)
 STATE['calls']+=1
 return self.main_forward(x,mode='native',include_bias=True)

class TrackedW8A8Linear(W8A8Linear):
 def forward(self,x):
  STATE['calls']+=1
  return super().forward(x)

def install(dit):
 make_h3_resident(dit)
 if STATE['variant']=='c5':
  for name in TARGET_NAMES:
   parent,attr=name.rsplit('.',1); old=dit.get_submodule(name); new=TrackedW8A8Linear(old); setattr(dit.get_submodule(parent),attr,new); old.weight=None; old.bias=None
  return {'main_linears':'symmetric K128 group W8A8 INT8 dot int32 accumulation, dynamic activation','target_count':200}
 weights={n:dit.get_submodule(n).weight.detach().cpu() for n in TARGET_NAMES}
 if STATE['variant'] in ('c3','c4'):
  info=install_plain_h3(dit,PLAIN,activation_packer=pack_activation_fast,verify_hashes=True)
  typ=PlainNativeH3Linear; fn=plain_forward
 else:
  info=install_native_h3(dit,EXPORT,activation_packer=pack_activation_fast,chunk_rows=1024,verify_hashes=True)
  typ=NativeH3Linear; fn=fused_forward
 for name in TARGET_NAMES:
  mod=dit.get_submodule(name);assert isinstance(mod,typ)
  mod.register_buffer('bf16_weight',weights[name]);mod._host_weights={'bf16_weight':weights[name], 'weight_packed':mod.weight_packed.cpu(), 'weight_scales_swizzled':mod.weight_scales_swizzled.cpu()};mod.forward=types.MethodType(fn,mod);STATE['modules'].append(mod)
 assert len(weights)==200
 torch.cuda.empty_cache()
 return {'main_linears':'native NVFP4 W4A4' if STATE['variant'] in ('c3','c4') else 'rank64 NVFP4 fused GEMM + BF16 low-rank down', 'bf16_endpoint_weights_resident':False,'weight_transfer_policy':'phase-level BF16/NVFP4 swap between CPU and GPU; included in step timings','export_manifest_sha256':info['manifest_sha256'],'target_count':200}

def switch_weights(step):
 if STATE['variant']=='c5':return
 endpoints={0,19} if STATE['variant']=='c2' else {0,1,18,19}
 target='bf16' if step in endpoints else 'quant'
 if target==STATE['precision']:return
 modules=STATE['modules']
 if target=='bf16':
  for m in modules:
   for k in ('weight_packed','weight_scales_swizzled'):m._buffers[k]=m._host_weights[k]
  torch.cuda.empty_cache()
  for m in modules:m._buffers['bf16_weight']=m._host_weights['bf16_weight'].to('cuda')
 else:
  for m in modules:m._buffers['bf16_weight']=m._host_weights['bf16_weight']
  torch.cuda.empty_cache()
  for m in modules:
   for k in ('weight_packed','weight_scales_swizzled'):m._buffers[k]=m._host_weights[k].to('cuda')
 STATE['precision']=target
 print(f'WEIGHT_PHASE step={step} mode={target}; CPU/GPU copy included',flush=True)

class Router(old.Router):
 def __init__(self,dit,arm):
  super().__init__(dit,'s70_fp4_20_int8_10')
  self.sparse=SparseConfig(query_block_size=128,sparsity_ratio=.75 if STATE['variant'] in ('c1','c4','c5') else .80,maxpool_weight=.2)
  self.dense=SparseConfig(query_block_size=128,sparsity_ratio=0.,maxpool_weight=.2)
  self.mixed=QuantConfig(nvfp4_ratio=.6 if STATE['variant'] in ('c1','c4','c5') else .5,int8_ratio=.4 if STATE['variant'] in ('c1','c4','c5') else .5,prefix_kv_precision='int8',prefix_query_precision='int8')
  self.int8=QuantConfig(nvfp4_ratio=0.,int8_ratio=1.,prefix_kv_precision='int8',prefix_query_precision='int8')
  self.manifest.update(schedule=STATE['variant'],video_tile_skip=self.sparse.sparsity_ratio,query_block_size=128,real_attention='Anemoi INT8 QK + FP8 PV and NVFP4',linear='plain NVFP4' if STATE['variant'] in ('c3','c4') else ('group128 W8A8' if STATE['variant']=='c5' else 'rank64 fused SVDQuant'))
 bind_layout=LayoutRouter.bind_layout
 @contextlib.contextmanager
 def forward_context(self,*,expected_cu,expected_refiner_cu,diagnostics=True):
  self.step+=1;STATE['step']=self.step;STATE['calls']=0;switch_weights(self.step)
  log=AttentionCallLog('bf16',expected_cu,expected_refiner_cu,diagnostics);token=self._active_log.set(log)
  try:
   with collect_fastpack_checks():yield log
   assert [r['block'] for r in log.main_rows]==list(range(50)) and log.refiner_helpers==2
   endpoint=self.step in ({0,19} if STATE['variant']=='c2' else {0,1,18,19})
   assert STATE['calls']==(200 if STATE['variant']=='c5' else (0 if endpoint else 200)),(self.step,STATE['calls'])
   expected_dense=50 if self.step<2 or (STATE['variant']=='c1' and self.step>=18) else (1 if STATE['variant'] in ('c1','c4','c5') else 0)
   assert sum(r['path']=='int8_dense' for r in log.main_rows)==expected_dense
   log.summary={'main_helper_calls':50,'refiner_helper_calls':2,'main_rows':log.main_rows,'linear_native_calls':STATE['calls']}
   if STATE['probe'] and self.step==2:raise ProbeComplete()
  finally:self._active_log.reset(token)
 def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
  log=self._active_log.get();block=self._active_main.get();expected=log.expected_refiner_cu if block is None else log.expected_cu
  assert q.shape==k.shape==v.shape==(expected[-1],56,128) and softmax_scale==128**-.5
  log.remember_cu(cu_seqlens,expected,'refiner' if block is None else 'main')
  if block is None:
   log.refiner_helpers+=1;return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
  n,total=expected[1:];dense=self.step<2 or (STATE['variant']=='c1' and (self.step>=18 or block==0)) or (STATE['variant'] in ('c4','c5') and block==0)
  if dense and STATE['variant'] in ('c4','c5'):
   out=self.original_helper(q,k,v,log.main_cpu_cu,softmax_scale);log.main_rows.append({'block':block,'path':'int8_dense','actual_path':'bf16_dense'});return out
  inputs=[t[:n].unsqueeze(0).contiguous() for t in (q,k,v)]
  result=anemoi_attention(*inputs,layout=VisualLayout(self.grid,self.prefix),layer=block,sparse_config=self.dense if dense else self.sparse,quant_config=self.int8 if dense else self.mixed,calibration=None if dense else old.calibration(block))
  out=torch.empty_like(q);out[:n]=result[0]
  if total>n:out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
  log.main_rows.append({'block':block,'path':'int8_dense' if dense else 'draft','prefix_tokens':self.prefix,'valid_length':n,'padding_length':total-n})
  return out

class SageRouter(sage.Router):
 def __init__(self,dit,arm):
  super().__init__(dit,'sage2pp' if STATE['variant']=='c6' else 'sage3_fixed');self.step=-1
 def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
  log=self._active_log.get();block=self._active_main.get();expected=log.expected_refiner_cu if block is None else log.expected_cu
  if STATE['variant']!='c7' or block is None or expected[1]<65536:
   return super()._dispatch(q,k,v,cu_seqlens,softmax_scale)
  n,total=expected[1:];log.remember_cu(cu_seqlens,expected,'main')
  parts=[]
  for start in range(0,56,8):
   Q,K,V=(x[:n,start:start+8].transpose(0,1).unsqueeze(0).contiguous() for x in (q,k,v))
   parts.append(sage.sageattn3_blackwell(Q,K,V,is_causal=False,per_block_mean=True)[0].transpose(0,1))
  out=torch.empty_like(q);out[:n]=torch.cat(parts,dim=1)
  out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
  log.main_rows.append({'block':block,'valid_length':n,'padding_length':total-n,'arm':'sage3_fixed','head_batch':8})
  log.fp4_calls+=1;log.original_bf16_segments+=1
  return out
 @contextlib.contextmanager
 def forward_context(self,**kw):
  self.step+=1;STATE['step']=self.step;STATE['calls']=0;switch_weights(self.step)
  with collect_fastpack_checks():
   with super().forward_context(**kw) as log:yield log
  assert STATE['calls']==(0 if self.step in {0,1,18,19} else 200)
  if STATE['probe'] and self.step==2:raise ProbeComplete()

def evaluate(pid):
 import av
 from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
 folder=base.REPORT/f'p{pid:03d}'; d=json.loads((folder/'bf16_denoise.json').read_text());v=json.loads((folder/'bf16_video.json').read_text())
 refroot=ROOT/('results/research/E124' if pid==16 else 'results/research/E150');refd=json.loads((refroot/f'p{pid:03d}/bf16_denoise.json').read_text());refv=json.loads((refroot/f'p{pid:03d}/bf16_video.json').read_text());height,width=d['settings']['height'],d['settings']['width'];assert d['initial_latent_sha256']==refd['initial_latent_sha256']
 def frames(path):
  with av.open(path) as c:
   for f in c.decode(video=0):yield f.to_ndarray(format='rgb24')
 metric=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True,reduction='none').cuda().eval();sse=energy=0;lp=[]
 with torch.inference_mode():
  for n,(r,x) in enumerate(zip_longest(frames(refv['video']),frames(v['video'])),1):
   assert r is not None and x is not None and r.shape==x.shape==(height,width,3)
   delta=x.astype(np.int32)-r.astype(np.int32);sse+=int(np.square(delta).sum(dtype=np.int64));energy+=int(np.square(r.astype(np.int32)).sum(dtype=np.int64))
   def t(z):return torch.from_numpy(z).permute(2,0,1).unsqueeze(0).cuda().float().div_(255)
   lp.append(float(metric(t(x),t(r))));metric.reset()
 assert n==d['frames'] and len(d['steps'])==20
 result={'status':'complete','case':pid,'variant':STATE['variant'],'frames':n,'reference':refv['video'],'video':v['video'],'rgb_nmse':sse/energy,'lpips_alex':float(np.mean(lp)),'psnr_db':10*math.log10(255**2*n*height*width*3/sse),'dit_seconds':sum(s['dit_seconds'] for s in d['steps']),'steps':d['steps'],'initial_latent_sha256':d['initial_latent_sha256']}
 (folder/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print('METRICS COMPLETE',json.dumps({k:v for k,v in result.items() if k!='steps'}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=('c1','c2','c3','c4','c5','c6','c7'),required=True);ap.add_argument('--pid',type=int,choices=(16,11,45,89),required=True);ap.add_argument('--phase',choices=('denoise','decode','evaluate','probe'),required=True);args=ap.parse_args()
 base.MANIFEST=ROOT/('results/research/h3_r64_draft_secondset/manifest_p016.json' if args.pid==16 else 'results/research/E150/manifest.json');STATE['variant']=args.variant;STATE['probe']=args.phase=='probe';base.REPORT=OUT/('probe_'+args.variant+os.environ.get('H3_PROBE_TAG','') if STATE['probe'] else args.variant);base.DATA=DATA/('probe_'+args.variant+os.environ.get('H3_PROBE_TAG','') if STATE['probe'] else args.variant);base.ARM=args.variant
 base.e121.Router=SageRouter if args.variant in ('c6','c7') else Router;base.e121.base.checked_linears=install
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
 if args.phase=='probe':
  try:base.denoise(args.pid)
  except ProbeComplete:
   result={'status':'complete','variant':args.variant,'pid':args.pid,'checked_steps':[0,1,2]};(base.REPORT/'probe_result.json').write_text(json.dumps(result,indent=2)+'\n');print('PROBE_COMPLETE',args.variant,flush=True)
  else:raise RuntimeError('probe did not stop at expected step')
 elif args.phase=='denoise':
  base.denoise(args.pid)
  path=base.REPORT/f'p{args.pid:03d}'/'bf16_denoise.json';data=json.loads(path.read_text());refroot=ROOT/('results/research/E124' if args.pid==16 else 'results/research/E150');ref=json.loads((refroot/f'p{args.pid:03d}/bf16_denoise.json').read_text());assert data['initial_latent_sha256']==ref['initial_latent_sha256'] and len(data['steps'])==20
  data['sum_dit_seconds']=sum(x['dit_seconds'] for x in data['steps']);data['initial_noise_matches_bf16']=True;data['precision_schedule']=args.variant;path.write_text(json.dumps(data,indent=2)+'\n')
 elif args.phase=='decode':base.decode(args.pid)
 else:evaluate(args.pid)
if __name__=='__main__':main()
