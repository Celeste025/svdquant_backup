#!/usr/bin/env python3
"""Reference fixed-scale encoder for FlashInfer 0.7.0.post1 SM120 attention.

This is diagnostic preparation, NOT an optimized attention quantizer. Native
fwd is unchanged. Input QKV are preprocessed exactly as the installed official
API; frozen scales must already be in its physical swizzled layout.
"""
from __future__ import annotations
import torch
from flashinfer.nvfp4_attention_sm120 import _preprocess_qkv


def logical_scales(buffer):
    """Undo native 64-row / 4-column scale swizzle, preserving leading axes."""
    *leading, rows, cols = buffer.shape
    if rows % 64 or cols % 4:
        raise ValueError(f'unsupported scale shape {tuple(buffer.shape)}')
    r=torch.arange(rows,device=buffer.device,dtype=torch.long)[:,None]
    c=torch.arange(cols,device=buffer.device,dtype=torch.long)[None,:]
    offsets=(r//64)*64*cols+(c//4)*256+(r%16)*16+((r%64)//16)*4+c%4
    return buffer.view(torch.uint8).reshape(-1,rows*cols)[:,offsets.flatten()].view(torch.float8_e4m3fn).reshape(*leading,rows,cols)


def k_permutation(length,device):
    t=torch.arange(length,device=device,dtype=torch.long);u=t%32
    return (t//32)*32+(u//8)*2+((u%8)//2)*8+u%2


def _encode_matrix(values,scale_buffer,chunk_rows=256):
    """Input and logical scale rows are already in K's physical permutation."""
    *leading,rows,cols=values.shape
    if tuple(scale_buffer.shape)!=(*leading,rows,cols//16):
        raise ValueError('scale/value shape mismatch')
    scales=logical_scales(scale_buffer).float().reshape(-1,cols//16)
    x=values.reshape(-1,cols)
    packed=torch.empty((x.shape[0],cols//2),dtype=torch.uint8,device=x.device)
    mids=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],dtype=torch.float32,device=x.device)
    counts=torch.zeros(5,dtype=torch.int64,device=x.device)
    clipped_energy=torch.zeros((),dtype=torch.float64,device=x.device)
    for begin in range(0,x.shape[0],chunk_rows):
        end=min(begin+chunk_rows,x.shape[0]);v=x[begin:end].float().reshape(end-begin,cols//16,16)
        s=scales[begin:end]
        reciprocal=torch.where(s==0,torch.zeros_like(s),s.reciprocal())
        z=v*reciprocal.unsqueeze(-1)
        az=z.abs().contiguous()
        lower=torch.bucketize(az,mids,right=False)
        exact_mid=(lower<7)&(az==mids[lower.clamp_max(6)])
        magnitude=lower+(exact_mid&((lower%2)==1)).to(torch.long)
        codes=(magnitude.to(torch.uint8)|(torch.signbit(z).to(torch.uint8)<<3)).reshape(end-begin,cols)
        packed[begin:end]=codes[:,0::2]|(codes[:,1::2]<<4)
        clipped=v.abs()>6*s.unsqueeze(-1)
        counts+=torch.stack([clipped.sum(),(s==0).sum(),((s==0)&(v.abs().amax(-1)>0)).sum(),
                             (magnitude==7).sum(),exact_mid.sum()])
        clipped_energy+=(v.double().square()*clipped).sum()
    vals=counts.cpu().tolist()
    stats=dict(zip(['clipped_elements','zero_scale_groups','nonzero_groups_with_zero_scale','max_magnitude_code_elements','exact_midpoints'],map(int,vals)))
    stats.update(elements=values.numel(),groups=scales.numel(),clipped_input_energy=clipped_energy.item(),
                 clip_fraction=vals[0]/values.numel(),zero_scale_fraction=vals[1]/scales.numel(),
                 max_code_is_not_clipping=True)
    return packed.reshape(*leading,rows,cols//2),stats


@torch.inference_mode()
def pack_with_scales(q,k,v,scale_pack,*,chunk_rows=256):
    """Re-encode CURRENT q/k/v under scales from a 7-tuple native pack.

    Returns (native-compatible seven-tensor tuple, per-QKV clipping stats).
    Means and correction are CURRENT, never copied from the scale donor.
    The caller must verify own-scale byte parity before causal interpretation.
    """
    if len(scale_pack)!=7:
        raise ValueError('expected FlashInfer native seven-tensor pack')
    qp,kp,vp,correction=_preprocess_qkv(q,k,v,per_block_mean=True)
    kp=kp.index_select(-2,k_permutation(kp.shape[-2],kp.device)).contiguous()
    vp=vp.transpose(-2,-1).contiguous()
    encoded=[];stats={}
    for name,values,sf in zip(('q','k','v'),(qp,kp,vp),scale_pack[3:6],strict=True):
        data,stats[name]=_encode_matrix(values,sf,chunk_rows=chunk_rows);encoded.append(data)
    packed=(*encoded,*scale_pack[3:6],correction)
    return packed,stats


def compare_packs(got,expected):
    if len(got)!=len(expected):raise ValueError('pack lengths differ')
    return {name:int(torch.count_nonzero(a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)).item())
            for name,a,b in zip(('q','k','v','q_scale','k_scale','v_scale','correction'),got,expected,strict=True)}
