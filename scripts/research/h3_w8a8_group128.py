"""Symmetric K128 group W8A8: true INT8 tensor-core GEMM with per-group scales."""
import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def _quant(X,Q,S,M:tl.constexpr,K:tl.constexpr,G:tl.constexpr):
 row=tl.program_id(0);group=tl.program_id(1);i=group*128+tl.arange(0,128)
 x=tl.load(X+row*K+i,i<K,0).to(tl.float32);scale=tl.maximum(tl.max(tl.abs(x),0)/127.,1.e-12)
 z=tl.extra.cuda.libdevice.nearbyint(x/scale);z=tl.minimum(127.,tl.maximum(-127.,z))
 tl.store(Q+row*K+i,z.to(tl.int8),i<K);tl.store(S+row*G+group,scale)

@triton.jit
def _gemm(A,W,AS,WS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,G:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64):
 mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);ki=tl.arange(0,128)
 acc=tl.full((BM,BN),0,tl.float32)
 for g in range(G):
  ka=g*128+ki
  aa=tl.load(A+mi[:,None]*K+ka[None,:],(mi[:,None]<M)&(ka[None,:]<K),0)
  ww=tl.load(W+ni[None,:]*K+ka[:,None],(ni[None,:]<N)&(ka[:,None]<K),0)
  i32=tl.dot(aa,ww,out_dtype=tl.int32)
  sa=tl.load(AS+mi*G+g,mi<M,0);sw=tl.load(WS+ni*G+g,ni<N,0)
  acc+=i32.to(tl.float32)*sa[:,None]*sw[None,:]
 tl.store(Y+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))

def quant(x):
 x=x.contiguous();m,k=x.shape;assert k%128==0
 q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((m,k//128),device=x.device,dtype=torch.float32)
 _quant[(m,k//128)](x,q,s,m,k,k//128,num_warps=4)
 return q,s

class W8A8Linear(nn.Module):
 def __init__(self,old):
  super().__init__();self.in_features=old.in_features;self.out_features=old.out_features
  assert self.in_features%128==0
  with torch.no_grad():q,s=quant(old.weight.detach())
  self.register_buffer('qw',q);self.register_buffer('sw',s);self.register_buffer('bias',old.bias.detach() if old.bias is not None else None)
 def forward(self,x):
  shape=x.shape;a=x.contiguous().view(-1,self.in_features);q,s=quant(a);m=a.shape[0];n=self.out_features
  out=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
  _gemm[(triton.cdiv(m,32),triton.cdiv(n,64))](q,self.qw,s,self.sw,out,m,n,self.in_features,self.in_features//128,num_warps=4)
  if self.bias is not None:out.add_(self.bias)
  return out.reshape(*shape[:-1],n)
