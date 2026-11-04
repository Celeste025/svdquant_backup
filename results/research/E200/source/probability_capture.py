"""Experimental Q128 INT8 probability-mean route capture, without full P.

Two QK passes: compute per-query global logsumexp, then recompute video
QK tiles and reduce normalized probabilities. This is not a fused mainloop.
"""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _lse(Q,K,QS,KS,VALID,LSE, QN:tl.constexpr,KN:tl.constexpr,D:tl.constexpr,
         QB:tl.constexpr,KB:tl.constexpr,H:tl.constexpr,SCALE:tl.constexpr,
         BM:tl.constexpr=32,BN:tl.constexpr=64):
    qchunk=tl.program_id(0);head=tl.program_id(1);batch=tl.program_id(2)
    rr=qchunk*BM+tl.arange(0,BM);dd=tl.arange(0,D);cc=tl.arange(0,BN)
    q=tl.load(Q+((batch*H+head)*QN+rr[:,None])*D+dd[None,:])
    qs=tl.load(QS+(batch*H+head)*QB+qchunk//4)
    m=tl.full((BM,),-float('inf'),tl.float32);l=tl.full((BM,),0.,tl.float32)
    for kb in range(KB):
        k=tl.load(K+((batch*H+head)*KN+(kb*BN+cc[None,:]))*D+dd[:,None])
        ks=tl.load(KS+(batch*H+head)*KB+kb)
        valid=tl.load(VALID+batch*KB+kb)
        s=tl.dot(q,k).to(tl.float32)*(qs*ks*SCALE)
        s=tl.where(cc[None,:]<valid,s,-float('inf'))
        mnew=tl.maximum(m,tl.max(s,1))
        old=tl.where(m==-float('inf'),0.,tl.exp(m-mnew))
        part=tl.where(cc[None,:]<valid,tl.exp(s-mnew[:,None]),0.)
        l=l*old+tl.sum(part,1);m=mnew
    tl.store(LSE+(batch*H+head)*QN+rr,m+tl.log(l))

@triton.jit
def _partials(Q,K,QS,KS,VALID,QC,LSE,OUT,QN:tl.constexpr,KN:tl.constexpr,D:tl.constexpr,
              QB:tl.constexpr,KB:tl.constexpr,VB:tl.constexpr,PB:tl.constexpr,H:tl.constexpr,
              SCALE:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64):
    qchunk=tl.program_id(0);vk=tl.program_id(1);bh=tl.program_id(2)
    batch=bh//H;head=bh%H;rr=qchunk*BM+tl.arange(0,BM);dd=tl.arange(0,D);cc=tl.arange(0,BN)
    qb=qchunk//4;qs=tl.load(QS+bh*QB+qb);nq=tl.load(QC+qb)
    q=tl.load(Q+(bh*QN+rr[:,None])*D+dd[None,:]);lse=tl.load(LSE+bh*QN+rr)
    total=tl.full((),0.,tl.float32)
    for half in range(2):
        kb=PB+2*vk+half;valid=tl.load(VALID+batch*KB+kb)
        kval=tl.load(K+(bh*KN+kb*BN+cc[None,:])*D+dd[:,None]);ks=tl.load(KS+bh*KB+kb)
        logits=tl.dot(q,kval).to(tl.float32)*(qs*ks*SCALE)
        p=tl.where((cc[None,:]<valid)&((rr%128)[:,None]<nq),tl.exp(logits-lse[:,None]),0.)
        total+=tl.sum(tl.sum(p,1),0)
    tl.store(OUT+((bh*QB+qb)*VB+vk)*4+qchunk%4,total)

@torch.inference_mode()
def int8_softmax_mean(int8_operands, valid_k, video_counts, prefix_blocks):
    q,qs,k,ks,_,_=int8_operands
    if q.ndim!=4 or k.ndim!=4 or q.shape[:2]!=k.shape[:2] or q.shape[-1] not in (64,128):
        raise ValueError('probability capture requires equal Q/K head counts and D=64 or 128')
    b,h,qn,d=q.shape;kn=k.shape[2];qb=qn//128;kb=kn//64;vb=video_counts.numel()
    if qn!=qb*128 or kb!=prefix_blocks+2*vb or qs.shape!=(b,h,qb) or ks.shape!=(b,h,kb):
        raise ValueError('probability capture requires Q128, K64 ragged layout')
    lse=torch.empty((b,h,qn),device=q.device,dtype=torch.float32)
    partial=torch.empty((b,h,qb,vb,4),device=q.device,dtype=torch.float32)
    _lse[(qn//32,h,b)](q,k,qs,ks,valid_k,lse,qn,kn,d,qb,kb,h,1/math.sqrt(d),num_warps=4)
    _partials[(qn//32,vb,b*h)](q,k,qs,ks,valid_k,video_counts,lse,partial,qn,kn,d,qb,kb,vb,prefix_blocks,h,1/math.sqrt(d),num_warps=4)
    # Native SM120 router accepts FP16; uniform rescaling preserves ranking
    # while reducing underflow for small probabilities.
    return (partial.sum(-1)/(video_counts.float().view(1,1,qb,1)*video_counts.float().view(1,1,1,vb))*1024.).to(torch.float16).contiguous()
