"""Native SM120 NVFP4 QK/PV MMA with V-guided FP8 PV tiles.

Experimental correctness-first kernel. FP4 uses E2M1 values and an E4M3
scale per 16 reduction elements through ``tl.dot_scaled``. The rare FP8 PV
route uses native FP8 ``tl.dot`` with one FP32 dynamic scale per 128 reduction
elements of each P row or V output channel.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

SAGE = Path("/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe")
sys.path.insert(0, str(SAGE))
from sageattn3.api import preprocess_qkv


@triton.jit
def _e2m1_code(x):
    ax = tl.abs(x)
    c = tl.full(x.shape, 0, tl.uint8)
    # Hardware cvt.rn.satfinite.e2m1x2.f32 rounds exact midpoints to
    # the code with an even low bit: 0 at .25, 1 at .75, 1 at 1.25, etc.
    c = tl.where(ax > 0.25, 1, c)
    c = tl.where(ax >= 0.75, 2, c)
    c = tl.where(ax > 1.25, 3, c)
    c = tl.where(ax >= 1.75, 4, c)
    c = tl.where(ax > 2.5, 5, c)
    c = tl.where(ax >= 3.5, 6, c)
    c = tl.where(ax > 5.0, 7, c)
    return (c | tl.where(x < 0, 8, 0).to(tl.uint8)).to(tl.uint8)


@triton.jit
def _native_kernel(Q, K, V, DS, O,
                   stride_qh, stride_ql, stride_kh, stride_kl,
                   stride_vh, stride_vl, stride_dsh, stride_dsb,
                   stride_oh, stride_ol, L: tl.constexpr, PAD: tl.constexpr,
                   SCALE: tl.constexpr, THRESHOLD_SQ: tl.constexpr):
    qb = tl.program_id(0)
    h = tl.program_id(1)
    r = qb * 128 + tl.arange(0, 128)
    d = tl.arange(0, 128)
    g = tl.arange(0, 8)
    pair = tl.arange(0, 2)

    q = tl.load(Q + h*stride_qh + r[:,None]*stride_ql + d[None,:]).to(tl.float32)
    qmax = tl.max(tl.abs(tl.reshape(q, (128,8,16))), axis=2)
    qs = (qmax / 6.).to(tl.float8e4nv)
    qsf = tl.broadcast_to(qs.to(tl.float32)[:,:,None], (128,8,16))
    qinv = tl.where(qsf == 0, 0., 1. / qsf)
    qc = _e2m1_code(tl.reshape(q,(128,8,16))*qinv)
    qc = tl.reshape(qc,(128,64,2))
    qp = tl.sum(qc * tl.where(pair[None,None,:] == 0, 1, 16).to(tl.uint8),axis=2).to(tl.uint8)

    m = tl.full((128,), -float('inf'), tl.float32)
    den = tl.full((128,), 0., tl.float32)
    acc = tl.full((128,128), 0., tl.float32)
    # Sage3's native online-softmax mainloop visits K/V tiles in reverse.
    for offset in range(0, PAD, 128):
        t = PAD - 128 - offset
        kr = t + tl.arange(0,128)
        k = tl.load(K + h*stride_kh + kr[:,None]*stride_kl + d[None,:]).to(tl.float32)
        km = tl.max(tl.abs(tl.reshape(k,(128,8,16))),axis=2)
        ks = (km/6.).to(tl.float8e4nv)
        ksf = tl.broadcast_to(ks.to(tl.float32)[:,:,None],(128,8,16))
        kinv = tl.where(ksf == 0, 0., 1. / ksf)
        kc = _e2m1_code(tl.reshape(k,(128,8,16))*kinv)
        kc = tl.reshape(kc,(128,64,2))
        kp = tl.sum(kc * tl.where(pair[None,None,:] == 0, 1, 16).to(tl.uint8),axis=2).to(tl.uint8)
        score = tl.dot_scaled(qp, qs, 'e2m1', tl.trans(kp), ks, 'e2m1')
        ds = tl.load(DS + h*stride_dsh + qb*stride_dsb + kr)
        score = score*SCALE + ds[None,:]
        m2 = tl.maximum(m,tl.max(score,axis=1))
        alpha = tl.exp(m-m2)
        p = tl.exp(score-m2[:,None])
        den = den*alpha + tl.sum(p,axis=1)
        acc = acc*alpha[:,None]

        v = tl.load(V + h*stride_vh + kr[:,None]*stride_vl + d[None,:]).to(tl.float32)
        vt = tl.trans(v)  # [D,K], group 16 along K
        vm = tl.max(tl.abs(tl.reshape(vt,(128,8,16))),axis=2)
        vs = (vm/6.).to(tl.float8e4nv)
        vsf = tl.broadcast_to(vs.to(tl.float32)[:,:,None],(128,8,16))
        vinv = tl.where(vsf == 0, 0., 1. / vsf)
        vc = _e2m1_code(tl.reshape(vt,(128,8,16))*vinv)
        vc = tl.reshape(vc,(128,64,2))
        vp = tl.sum(vc * tl.where(pair[None,None,:] == 0, 1, 16).to(tl.uint8),axis=2).to(tl.uint8)
        # Tile decision is based on V FP4 dequantization error.
        code_mag = vc & 7
        val = tl.where(code_mag == 0,0.,tl.where(code_mag == 1,.5,
              tl.where(code_mag == 2,1.,tl.where(code_mag == 3,1.5,
              tl.where(code_mag == 4,2.,tl.where(code_mag == 5,3.,
              tl.where(code_mag == 6,4.,6.)))))))
        vq = tl.reshape(val * tl.where((vc & 8) != 0,-1.,1.),(128,8,16))*vsf
        vdiff = tl.reshape(vq,(128,128))-vt
        fp4err = tl.sum(vdiff*vdiff)
        energy = tl.sum(vt*vt)
        if fp4err < THRESHOLD_SQ*energy:
            pm = tl.max(tl.reshape(p,(128,8,16)),axis=2)
            ps = (pm*448.).to(tl.float8e4nv)
            pden = tl.broadcast_to(tl.where(pm == 0,1.,pm)[:,:,None],(128,8,16))
            pc = _e2m1_code(tl.reshape(p,(128,8,16))/pden*6.)
            pc = tl.reshape(pc,(128,64,2))
            pp = tl.sum(pc * tl.where(pair[None,None,:] == 0, 1, 16).to(tl.uint8),axis=2).to(tl.uint8)
            pv = tl.dot_scaled(pp,ps,'e2m1',tl.trans(vp),vs,'e2m1') / 2688.
        else:
            # One FP32 dynamic scale per full 128-key row/channel. This is
            # matched by the PyTorch reference with fp8 group=128.
            p8s = tl.maximum(tl.max(p,axis=1)/448.,1e-12)
            v8s = tl.maximum(tl.max(tl.abs(vt),axis=1)/448.,1e-12)
            p8 = tl.minimum(p/p8s[:,None],448.).to(tl.float8e4nv)
            v8 = tl.maximum(tl.minimum(vt/v8s[:,None],448.),-448.).to(tl.float8e4nv)
            pv = tl.dot(p8,tl.trans(v8),out_dtype=tl.float32)*p8s[:,None]*v8s[None,:]
        acc += pv
        m = m2
    out = acc/den[:,None]
    tl.store(O + h*stride_oh + r[:,None]*stride_ol + d[None,:],out,mask=r[:,None]<L)


def native_mixed_sage3_attention(q: torch.Tensor,k: torch.Tensor,v: torch.Tensor,
                                 *,threshold: float=.10,
                                 match_sage3_padding_mask: bool=False):
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError('expected matching [1,H,L,128]')
    if q.dtype != torch.bfloat16 or not q.is_cuda or not 0 < threshold < 1:
        raise ValueError('expected CUDA BF16 and threshold in (0,1)')
    qp,kp,vp,ds=preprocess_qkv(q,k.clone(),v,per_block_mean=True)
    _,heads,pad,dim=qp.shape
    length=q.shape[2]
    ds=ds[0].float().contiguous()
    if pad>length:
        if match_sage3_padding_mask:
            # Official Sage3 masks accumulator columns in K's physical
            # within-32 permutation. The actual score columns correspond to
            # logical keys below; preserving this behavior is only for
            # numerical comparisons with that kernel, not a corrected mask.
            physical=torch.arange(length,pad,device=q.device)
            u=physical%32
            logical=(physical//32)*32+(u//8)*2+((u%8)//2)*8+u%2
            ds[:,:,logical]=-float('inf')
        else:
            ds[:,:,length:]=-float('inf')
    ds.mul_(dim**-.5)
    out=torch.empty((heads,pad,dim),device=q.device,dtype=torch.float32)
    launch=_native_kernel[(pad//128,heads)](
        qp[0],kp[0],vp[0],ds,out,
        qp.stride(1),qp.stride(2),kp.stride(1),kp.stride(2),
        vp.stride(1),vp.stride(2),ds.stride(0),ds.stride(1),
        out.stride(0),out.stride(1),L=length,PAD=pad,
        SCALE=dim**-.5,THRESHOLD_SQ=threshold*threshold,num_stages=1)
    ptx=launch.asm['ptx']
    native_mixed_sage3_attention.last_mma_check={
        'nvfp4_block16_mma': 'mxf4nvf4.block_scale.scale_vec::4X' in ptx,
        'fp8_mma': '.f8f6f4.' in ptx or '.e4m3.e4m3' in ptx,
    }
    if not all(native_mixed_sage3_attention.last_mma_check.values()):
        raise RuntimeError(f'native mixed GEMM PTX check failed: {native_mixed_sage3_attention.last_mma_check}')
    return out[:,:length].unsqueeze(0).to(torch.bfloat16).contiguous()
