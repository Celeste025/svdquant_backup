"""E082 private native center128 route; original Sage3 files stay untouched."""
from pathlib import Path
import torch
from probe_h3_sage3_interaction import Router as Original
from probe_h3_plain_baseline import file_record
import e082_sage_center as center

def contract():
 return dict(private=center.contract(),router=file_record(__file__),group=128,video_tokens=21312,scope='full video-only key groups; all valid queries, original refiner/padding')

class Router(Original):
 def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
  log=self._active_log.get();block=self._active_main.get()
  if block is None:return super()._dispatch(q,k,v,cu_seqlens,softmax_scale)
  assert log is not None and softmax_scale==128**-.5
  log.remember_cu(cu_seqlens,log.expected_cu,'main')
  n,total=log.expected_cu[1:];assert q.shape==k.shape==v.shape==(total,56,128)
  assert n==21731 and all(x.dtype==torch.bfloat16 for x in (q,k,v))
  Q,K,V=[x[:n].transpose(0,1).unsqueeze(0).contiguous() for x in (q,k,v)]
  result=center.sageattn3_center128(Q,K.clone(),V,n-21312,center=True)
  assert result.shape==Q.shape and result.dtype==q.dtype
  out=torch.empty_like(q);out[:n]=result[0].transpose(0,1)
  out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
  log.main_rows.append(dict(block=block,valid_length=n,padding_length=total-n,center_group=128))
  log.fp4_calls+=1;log.original_bf16_segments+=1;log.finite(out)
  return out
