#!/usr/bin/env python3
"""Meaningful GPU parity checks against independently implemented native conversion."""
import hashlib,json,traceback
from pathlib import Path
import torch
from nvfp4_attention_fixed_scale import pack_with_scales,compare_packs,_encode_matrix
from flashinfer.nvfp4_attention_sm120 import nvfp4_attention_sm120_quantize_qkv as quant
from flashinfer.nvfp4_attention_sm120 import nvfp4_attention_sm120_fwd as run
from flashinfer.nvfp4_attention_sm120 import get_nvfp4_attention_sm120_module

torch.set_num_threads(4);torch.manual_seed(20261002);torch.backends.cuda.matmul.allow_tf32=False
report={'purpose':'fixed_scale_packer_byte_parity','rows':[],
        'helper_sha256':hashlib.sha256(Path(__file__).with_name('nvfp4_attention_fixed_scale.py').read_bytes()).hexdigest()}
p=Path('results/research/E006_fixed_packer_parity.json')
try:
 with torch.inference_mode():
  for n,zero in [(128,False),(257,False),(16,True),(129,False)]:
   shape=(2,2,n,128);q,k,v=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
   if zero:q.zero_();k.zero_();v.zero_()
   native=quant(q,k,v);own,stats=pack_with_scales(q,k,v,native);diff=compare_packs(own,native)
   row={'n':n,'zero':zero,'different_bytes':diff,'stats':stats};report['rows'].append(row)
   print(json.dumps(row),flush=True)
   if any(diff.values()):raise AssertionError(diff)
   a=run(*native,return_lse=False,unpadded_k_len=n);b=run(*own,return_lse=False,unpadded_k_len=n)
   assert torch.equal(a,b)
  vals=[-6,-5,-3.5,-2.5,-1.75,-1.25,-.75,-.25,.25,.75,1.25,1.75,2.5,3.5,5,6]
  for name,vector in [('exact_midpoints',vals),('signed_zero',[-0.,0.]*7+[-6.,6.]),('scale_underflow',[-1e-8,1e-8]*8)]:
   x=torch.tensor(vector,device='cuda',dtype=torch.bfloat16).repeat(1,1,128,8).contiguous()
   pk=torch.empty((1,1,128,64),device='cuda',dtype=torch.uint8)
   sf=torch.empty((1,1,128,8),device='cuda',dtype=torch.float8_e4m3fn)
   get_nvfp4_attention_sm120_module().scaled_fp4_quant(x,pk,sf,1)
   ours,stats=_encode_matrix(x,sf)
   diff=int(torch.count_nonzero(ours!=pk));report['rows'].append({'case':name,'different_bytes':diff,'stats':stats})
   print(name,diff,stats,flush=True)
   if diff:raise AssertionError((name,diff))
 report['status']='passed'
except Exception:
 report['status']='failed';report['error']=traceback.format_exc()
 p.write_text(json.dumps(report,indent=2)+'\n');raise
p.write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],flush=True)
