"""Experimental native dense INT8 capture + logsumexp reduction, Q128 D128."""
from functools import lru_cache
from pathlib import Path
import torch
import triton
import triton.language as tl

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parents[4]
    return load(name='anemoi_probability_capture_fused',sources=[str(root/'csrc/attention/cuda/sm120/probability_capture_fused.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-include','cassert'],verbose=False)

@triton.jit
def _reduce(Local,LSE,Counts,Out,QB:tl.constexpr,KB:tl.constexpr,PB:tl.constexpr):
    qb=tl.program_id(0);kb=tl.program_id(1);bh=tl.program_id(2)
    r=tl.arange(0,128);half=tl.arange(0,2)
    x=tl.load(Local+(((bh*QB+qb)*KB+PB+kb*2+half[None,:])*128+r[:,None]))
    lse=tl.load(LSE+bh*QB*128+qb*128+r)
    nq=tl.load(Counts+qb);nk=tl.load(Counts+kb)
    p=tl.where(r[:,None]<nq,tl.exp(x-lse[:,None]),0.)
    value=tl.sum(tl.sum(p,1),0)/(nq*nk)*1024.
    tl.store(Out+(bh*QB+qb)*QB+kb,value)

def fused_softmax_mean(operands,valid,counts,prefix_blocks):
    q,qs,k,ks,v,vs=operands
    if q.size(-1)!=128:raise ValueError('fused probability capture currently requires D128')
    output,local,lse=extension().capture(q,qs,k,ks,v,vs,valid)
    b,h,qn,_=q.shape;qb=qn//128;kb=k.size(2)//64
    score=torch.empty((b,h,qb,qb),device=q.device,dtype=torch.float16)
    _reduce[(qb,qb,b*h)](local,lse,counts,score,qb,kb,prefix_blocks,num_warps=4)
    return output,score
