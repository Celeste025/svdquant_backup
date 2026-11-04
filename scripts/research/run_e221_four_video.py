#!/usr/bin/env python3
"""E150 paired rollouts: protected baseline and Q64 probability refresh."""
import argparse, json
from contextlib import contextmanager
from pathlib import Path
import torch
import run_e159_route_reuse_whitebox as prior
import run_e150_whitebox_bf16 as run
from anemoi import AttentionRouteCache, SparseConfig, QuantConfig, VisualLayout, anemoi_attention

ROOT=Path(__file__).resolve().parents[2]
class Router(prior.Router):
 def __init__(self,dit,arm):
  super().__init__(dit,arm)
  self.policy=POLICY
  self.dense_first_steps=2
  self.is_refresh=POLICY in ('refresh64','sparse_refresh64')
  self.is_sparse=POLICY in ('sparse128','sparse_refresh64')
  self.dense_first_layers=1 if self.is_refresh else 0
  self.sparse=SparseConfig(query_block_size=128 if POLICY=='sparse128' else 64,sparsity_ratio=.75 if self.is_sparse else 0.,maxpool_weight=.2)
  self.quant=QuantConfig(nvfp4_ratio=.6 if self.is_sparse else .9,int8_ratio=.4 if self.is_sparse else .1,prefix_kv_precision='int8',prefix_query_precision='int8')
  self.dense=QuantConfig(nvfp4_ratio=0.,int8_ratio=1.,prefix_kv_precision='int8',prefix_query_precision='int8')
  self.last_sparse=SparseConfig(query_block_size=64,sparsity_ratio=.8,maxpool_weight=.2)
  self.last_quant=QuantConfig(nvfp4_ratio=0.,int8_ratio=1.,prefix_kv_precision='int8',prefix_query_precision='int8')
  self.cache=AttentionRouteCache(capture_mode='softmax_mean_fused_k64')
  self.manifest.update(policy=POLICY,sparse_config=vars(self.sparse),quant_config=vars(self.quant),dense_first_steps=2,dense_first_layers=1 if self.is_refresh else 0,refresh_steps_1based=[2,10] if self.is_refresh else [],last_block='step 3 onward: 80% skip, 20% INT8, original DraftMap' if self.is_refresh else 'baseline',capture_mode='softmax_mean_fused_k64' if self.is_refresh else None,linears='BF16')
 @contextmanager
 def forward_context(self,**kwargs):
  with super(prior.Router,self).forward_context(**kwargs) as log:
   yield log
  if self.is_refresh:
   count=sum(bool(r['capture']) for r in log.main_rows)
   assert count==(48 if self.step in (1,9) else 0),(self.step,count)
   assert len(self.cache)==(0 if self.step==0 else 48),(self.step,len(self.cache))
   self.manifest.update(cached_routes=len(self.cache),cached_route_bytes=self.cache.allocated_bytes)
 def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
  log=self._active_log.get();block=self._active_main.get()
  expected=log.expected_refiner_cu if block is None else log.expected_cu
  assert softmax_scale==128**-.5 and q.shape==k.shape==v.shape==(expected[-1],56,128)
  log.remember_cu(cu_seqlens,expected,'refiner' if block is None else 'main')
  if block is None:
   log.refiner_helpers+=1
   return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
  n,total=expected[1:]; capture=self.is_refresh and self.step in (1,9) and 1<=block<49
  if self.is_refresh and block==0:
   out=self.original_helper(q,k,v,log.main_cpu_cu,softmax_scale);path='bf16'
  else:
   tensors=[t[:n].unsqueeze(0).contiguous() for t in (q,k,v)]
   args=dict(layout=VisualLayout(self.grid,self.prefix),layer=block,sparse_config=self.sparse,quant_config=self.quant,calibration=prior.prior.calibration(block))
   if self.is_refresh and block<49 and self.step>=1:
    if capture:
     result=anemoi_attention(*tensors,**args,first_step=True,route_cache=self.cache);path='dense_capture'
    else:
     result=anemoi_attention(*tensors,**args,first_step=False,route_cache=self.cache);path='mixed_reuse'
   else:
    if self.step<2:
     args['quant_config']=self.dense;args['sparse_config']=SparseConfig(query_block_size=self.sparse.query_block_size,sparsity_ratio=0.,maxpool_weight=.2);path='dense_int8'
    elif self.is_refresh and block==49: args['sparse_config']=self.last_sparse;args['quant_config']=self.last_quant;path='last_draft20'
    else: path='draft_mixed'
    result=anemoi_attention(*tensors,**args)
   out=torch.empty_like(q);out[:n]=result[0]
   if total>n:out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
  log.main_rows.append(dict(block=block,path='dense' if self.step<2 or (self.is_refresh and block==0) else 'draft',detail=path,capture=capture,prefix_tokens=self.prefix,valid_length=n,padding_length=total-n))
  return out

def main():
 global POLICY
 p=argparse.ArgumentParser();p.add_argument('--prompt-id',type=int,choices=(25,48,44,7),required=True);p.add_argument('--policy',choices=('refresh64','sparse128','sparse_refresh64'),required=True);p.add_argument('--phase',choices=('denoise','decode'),required=True);a=p.parse_args();POLICY=a.policy;prior.MODE='proxy'
 ref=json.loads((ROOT/f'results/research/E150/p{a.prompt_id:03d}/bf16_denoise.json').read_text())
 run.REPORT=ROOT/'results/research/E221'/POLICY;run.DATA=ROOT/'results/local/E221'/POLICY;run.ARM=POLICY;run.e121.Router=Router
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 read=run.read_prompt;run.read_prompt=lambda pid:read(pid,Path('/home/wjq/workspace/178866172854036'))
 if a.phase=='decode':run.decode(a.prompt_id);return
 run.denoise(a.prompt_id)
 path=run.REPORT/f'p{a.prompt_id:03d}/bf16_denoise.json';d=json.loads(path.read_text())
 assert all(d[k]==ref[k] for k in ('prompt','seed','settings','frames','initial_latent_sha256'))
 assert len(d['steps'])==20 and all(s['attention_main_calls']==50 for s in d['steps'])
 x=torch.load(d['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].float();y=torch.load(ref['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].float()
 d['comparison']={'bf16_reference':ref['terminal_latents'],'video_latent_nmse':float((x-y).double().square().sum()/y.double().square().sum()),'initial_noise_matches_bf16':True};path.write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
