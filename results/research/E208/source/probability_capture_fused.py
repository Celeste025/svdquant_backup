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

@lru_cache(None)
def rectangular_extension():
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parents[4]
    return load(name='anemoi_route_rect',sources=[str(root/'csrc/attention/cuda/sm120/h3_route_rect.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-include','cassert'],verbose=False)

@triton.jit
def _reduce_k64(Local,LSE,QC,KC,Out,QB:tl.constexpr,KB:tl.constexpr,PB:tl.constexpr,QSZ:tl.constexpr=128,KFACTOR:tl.constexpr=2):
    qb=tl.program_id(0);col=tl.program_id(1);bh=tl.program_id(2)
    r=tl.arange(0,QSZ)
    x=tl.load(Local+((bh*QB+qb)*KB+PB+col)*QSZ+r)
    lse=tl.load(LSE+bh*QB*QSZ+qb*QSZ+r)
    nq=tl.load(QC+qb);nk=tl.load(KC+PB+col)
    mass=tl.sum(tl.where(r<nq,tl.exp(x-lse),0.),0)
    value=tl.where(nk>0,mass/(nq*tl.maximum(nk,1))*1024.,0.)
    tl.store(Out+(bh*QB+qb)*(KFACTOR*QB)+col,value)

def fused_softmax_mean_k64(operands,valid,counts,prefix_blocks,query_block=128):
    q,qs,k,ks,v,vs=operands
    output,local,lse=(extension() if query_block==128 else q64_extension()).capture(q,qs,k,ks,v,vs,valid)
    b,h,qn,_=q.shape;qb=qn//query_block;kb=k.size(2)//64
    factor=query_block//64
    score=torch.empty((b,h,qb,factor*qb),device=q.device,dtype=torch.float16)
    _reduce_k64[(qb,factor*qb,b*h)](local,lse,counts,valid,score,qb,kb,prefix_blocks,query_block,factor,num_warps=4)
    # Empty physical K64 halves carry no work and must not consume budget.
    columns=torch.nonzero(valid[0,prefix_blocks:]>0).flatten()
    return output,score.index_select(-1,columns).contiguous(),columns

def route_k64(score,columns,sparsity,ratios):
    from .routing import _route_counts,RaggedRoutePlan
    n16,n8,n4=_route_counts(score.size(-2)*score.size(-1),sparsity_ratio=sparsity,nvfp4_ratio=ratios[0],fp8_ratio=ratios[1]+ratios[2],fp16_ratio=ratios[3])
    ids,c4,c8,c16=rectangular_extension().route(score,n16,n8,n4,None,None,0)
    # IDs name physical video K64 tiles; prefix is inserted by materialization.
    ids=columns[ids.long()].to(torch.int32).contiguous()
    return RaggedRoutePlan(ids,c4,c8,c16,n16,n8,n4)

@lru_cache(None)
def q64_extension():
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parents[4]
    return load(name='anemoi_probability_capture_q64',sources=[str(root/'csrc/attention/cuda/sm120/probability_capture_q64.cu')],extra_cuda_cflags=['-O3','--use_fast_math','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-include','cassert'],verbose=False)
