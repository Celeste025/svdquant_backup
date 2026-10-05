"""E020 main-weight QAD and plain native export for the 300 Wan block linears.

Only weight_master is trainable. This uses the frozen legacy-Wan dynamic NVFP4
recipe, not hardware E2M1 RNE: E4M3 midpoint ties go upward and E2M1 ties choose
the larger signed value. No smoothing, low-rank branch, or calibration is added.
GPU arithmetic/gradient/export checks belong to the bounded E020 runner.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl
import wan_nvfp4_fastpack as fast
from wan_native_nvfp4 import PackedNVFP4, NativeWanLinear

RECIPE = {
    'name': 'legacy_wan_dynamic_nvfp4_mainweight_qad', 'group_size': 16,
    'global': 'FP32: (amax * (1/6)) * (1/448), all-zero canonical global=1',
    'block_scale': 'E4M3 ties toward larger value',
    'code': 'E2M1 ties toward larger signed value',
    'effective_zero': 'legacy remove_zero=1; reject zero-SF with nonzero codes',
    'weight_input': 'FP32 master', 'activation_input': 'BF16',
    'qdq_output': 'BF16', 'ste_backward': 'identity cast to input dtype',
    'smoothing': False, 'low_rank': False,
}
SUFFIXES = ('attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0',
            'attn2.to_q','attn2.to_k','attn2.to_v','attn2.to_out.0',
            'ffn.net.0.proj','ffn.net.2')
TARGET_NAMES = tuple(f'blocks.{i}.{suffix}' for i in range(30) for suffix in SUFFIXES)
# Reuse exactly the collector used by deployment activation packing. Both W/A
# QDQ append flags, so an ordinary full training forward contributes 600 flags.
collect_qad_checks = fast.collect_fastpack_checks


def pack_nvfp4(x: torch.Tensor) -> PackedNVFP4:
    """Frozen three-kernel packer with FP32-master support; no input BF16 cast.

    Call under collect_qad_checks() to defer validation until scope exit. Outside
    that context each call synchronously rejects invalid/unrepresentable inputs.
    """
    if x.dtype not in (torch.bfloat16,torch.float32) or not x.is_cuda or not x.is_contiguous():
        raise ValueError('Expected contiguous CUDA BF16 activation or FP32 master')
    if x.ndim < 2 or not x.numel() or x.shape[-1] % 32:
        raise ValueError('Expected nonempty matrix/batched matrix, K divisible by 32')
    k=x.shape[-1]; m=x.numel()//k
    cp=triton.cdiv(k//16,4)*4; rp=triton.cdiv(m,128)*128
    count=triton.cdiv(x.numel(),16384)
    partial=torch.empty(count,device=x.device,dtype=torch.float32)
    global_scale=torch.empty(1,device=x.device,dtype=torch.float32)
    flags=torch.empty(2,device=x.device,dtype=torch.int32)
    codes=torch.empty((m,k//2),device=x.device,dtype=torch.uint8)
    scales=torch.empty(rp*cp,device=x.device,dtype=torch.float8_e4m3fn)
    fast._amax_parts[(count,)](x,partial,x.numel(),16384,num_warps=8,enable_fp_fusion=False)
    fast._global_scale[(1,)](partial,global_scale,flags,count,triton.next_power_of_2(count),enable_fp_fusion=False)
    fast._encode_groups[(triton.cdiv(rp*cp,128),)](
        x,global_scale,codes,scales,flags,m,k,cp,rp,128,num_warps=4,enable_fp_fusion=False)
    checks=fast._CHECKS.get()
    if checks is None: fast.validate_flags([flags])
    else: checks.append(flags)
    return PackedNVFP4(codes,None,global_scale,scales,tuple(x.shape),RECIPE['name'])


@triton.jit
def _decode_bf16(Codes,Scales,Global,Y,N:tl.constexpr,K:tl.constexpr,CP:tl.constexpr,BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    r=ix//K; c=(ix%K)//16
    packed=tl.load(Codes+ix//2,ix<N,0).to(tl.int32)
    code=(packed>>((ix%2)*4))&15
    magnitude=code&7
    value=tl.where(magnitude<4,magnitude.to(tl.float32)*0.5,
                   tl.where(magnitude==4,2.,tl.where(magnitude==5,3.,tl.where(magnitude==6,4.,6.))))
    value=tl.where((code&8)!=0,-value,value)
    sf_offset=(r//128)*128*CP+(c//4)*512+(r%32)*16+((r%128)//32)*4+c%4
    sf=tl.load(Scales+sf_offset,ix<N,0).to(tl.float32)
    effective=sf*tl.load(Global)
    # Match legacy dequantization, including its remove_zero convention.
    effective=tl.where(effective==0,1.,effective)
    tl.store(Y+ix,value*effective,ix<N)


def decode_packed(packet: PackedNVFP4) -> torch.Tensor:
    """One Triton launch, actual FP32 code*(SF*global) rounded to BF16."""
    if not packet.packed.is_cuda:
        raise ValueError('Single-kernel decoding requires CUDA packets')
    rows,k2=packet.packed.shape; k=k2*2
    out=torch.empty(packet.original_shape,device=packet.packed.device,dtype=torch.bfloat16)
    _decode_bf16[(triton.cdiv(rows*k,1024),)](
        packet.packed,packet.swizzled_scales,packet.global_scale,out,
        rows*k,k,triton.cdiv(k//16,4)*4,1024,num_warps=4,enable_fp_fusion=False)
    return out


class _QDQIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.input_dtype=x.dtype
        return decode_packed(pack_nvfp4(x.contiguous()))

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output.to(ctx.input_dtype)


def qdq_ste(x: torch.Tensor) -> torch.Tensor:
    """Return actual QDQ values; no x + (Q(x)-x) cancellation in forward."""
    return _QDQIdentity.apply(x)


class QADLinear(nn.Module):
    def __init__(self,original:nn.Linear):
        super().__init__()
        if type(original) is not nn.Linear or original.weight.device.type=='meta':
            raise ValueError('QADLinear needs a materialized, unquantized nn.Linear')
        if original.weight.dtype!=torch.bfloat16:
            raise ValueError('Original teacher weight must be BF16')
        if original.in_features%32 or original.out_features%32:
            raise ValueError('Native weight dimensions must be divisible by 32')
        if original._forward_pre_hooks or original._forward_hooks:
            raise ValueError('Do not wrap an already smoothed/quantized/hooked Linear')
        self.in_features=original.in_features; self.out_features=original.out_features
        self.weight_master=nn.Parameter(original.weight.detach().float().clone())
        self.register_buffer('bias',None if original.bias is None else original.bias.detach().to(torch.bfloat16).clone())
        self.train(original.training)

    def forward(self,x):
        if x.dtype!=torch.bfloat16:
            raise ValueError('QAD activation input must be BF16')
        return F.linear(qdq_ste(x),qdq_ste(self.weight_master),self.bias)


def _targets(model,allowed):
    if len(model.blocks)!=30:
        raise ValueError('Expected the fixed 30-block Wan transformer')
    found={name:module for name,module in model.named_modules()
           if name.startswith('blocks.') and isinstance(module,allowed)}
    if set(found)!=set(TARGET_NAMES):
        raise ValueError(f'Expected exactly the original 300 linears; found {len(found)}')
    # Ancestor hooks can hide cross-attention smoothing or other old transforms.
    if any(m._forward_pre_hooks or m._forward_hooks for m in model.modules()):
        raise ValueError('Plain main-weight baseline requires an unhooked model')
    return found


def _replace(model,name,module):
    parent,leaf=name.rsplit('.',1)
    setattr(model.get_submodule(parent),leaf,module)


def install_qad(model:nn.Module) -> dict[str,QADLinear]:
    """Freeze all other parameters and replace exactly 30*10 original linears."""
    targets=_targets(model,nn.Linear)
    for p in model.parameters(): p.requires_grad_(False)
    installed={}
    for name in TARGET_NAMES:
        installed[name]=QADLinear(targets[name])
        _replace(model,name,installed[name])
    return installed


@torch.no_grad()
def export_packed(model:nn.Module) -> dict:
    """CPU-only artifact contents; the packing operation itself runs on CUDA.

    Export current FP32 masters directly. No BF16 master rounding, new scale
    calibration, smoothing, or residual subtraction is performed.
    """
    targets=_targets(model,QADLinear)
    layers={}
    with collect_qad_checks():
        for name in TARGET_NAMES:
            layer=targets[name]; packet=pack_nvfp4(layer.weight_master.detach().contiguous())
            layers[name]=dict(packed=packet.packed.cpu(),scales=packet.swizzled_scales.cpu(),
                global_scale=packet.global_scale.cpu(),bias=None if layer.bias is None else layer.bias.detach().cpu().clone(),
                shape=[layer.out_features,layer.in_features])
    target_prefixes=tuple(name+'.' for name in TARGET_NAMES)
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()
           if not name.startswith(target_prefixes)}
    return dict(format_version=1,recipe=dict(RECIPE),target_count=300,layers=layers,non_target_state=state)


class PlainPackedWanLinear(NativeWanLinear):
    """Plain deployment: packed W only, no BF16 W/FP32 master/LR or hooks."""
    def __init__(self,packet,bias):
        super().__init__(packet,bias,None,activation_packer=fast.pack_activation_fast)
        self.native_calls=0

    def main_from_packet(self,activation,*,mode='native',include_bias=False):
        if mode=='native': self.native_calls+=1
        return super().main_from_packet(activation,mode=mode,include_bias=include_bias)


@torch.no_grad()
def install_packed(model:nn.Module,artifact:dict) -> dict[str,PlainPackedWanLinear]:
    """Install on a fresh BF16 or QAD model; restore non-target state as saved.

    Old target modules are released from the graph. The trainer must separately
    discard its optimizer/parameter-dictionary references before measuring
    deployment memory; they otherwise retain masters outside the model.
    """
    if artifact.get('format_version')!=1 or artifact.get('recipe')!=RECIPE or artifact.get('target_count')!=300:
        raise ValueError('Packed artifact recipe/version mismatch')
    if set(artifact['layers'])!=set(TARGET_NAMES):
        raise ValueError('Packed artifact must contain all 300 exact target names')
    targets=_targets(model,(nn.Linear,QADLinear))
    prefixes=tuple(name+'.' for name in TARGET_NAMES)
    expected={name for name in model.state_dict() if not name.startswith(prefixes)}
    if set(artifact['non_target_state'])!=expected:
        raise ValueError('Non-target state coverage changed')
    installed={}
    for name in TARGET_NAMES:
        original=targets[name]; row=artifact['layers'][name]
        device=(original.weight_master if isinstance(original,QADLinear) else original.weight).device
        n,k=original.out_features,original.in_features
        if row['shape']!=[n,k] or row['packed'].shape!=(n,k//2) or row['packed'].dtype!=torch.uint8:
            raise ValueError(f'Weight packet shape/dtype mismatch: {name}')
        if row['scales'].dtype!=torch.float8_e4m3fn or row['scales'].numel()!=triton.cdiv(n,128)*128*triton.cdiv(k//16,4)*4:
            raise ValueError(f'Weight scale shape/dtype mismatch: {name}')
        if row['global_scale'].shape!=(1,) or row['global_scale'].dtype!=torch.float32:
            raise ValueError(f'Weight global shape/dtype mismatch: {name}')
        packet=PackedNVFP4(row['packed'].to(device),None,row['global_scale'].to(device),
            row['scales'].to(device),(n,k),RECIPE['name'])
        bias=None if row['bias'] is None else row['bias'].to(device=device,dtype=torch.bfloat16)
        installed[name]=PlainPackedWanLinear(packet,bias).train(original.training)
        _replace(model,name,installed[name])
    loaded=model.load_state_dict(artifact['non_target_state'],strict=False)
    if loaded.unexpected_keys or any(not key.startswith(prefixes) for key in loaded.missing_keys):
        raise ValueError('Non-target state restoration mismatch')
    for p in model.parameters(): p.requires_grad_(False)
    return installed
