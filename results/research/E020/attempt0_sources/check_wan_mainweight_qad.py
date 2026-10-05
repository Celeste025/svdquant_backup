#!/usr/bin/env python3
"""Meaningful GPU checks of STE, FP32-master packing and real NVFP4 GEMM."""
import hashlib,json,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import wan_mainweight_qad as q
from wan_nvfp4_fastpack import pack_legacy_wan,collect_fastpack_checks
ROOT=Path(__file__).resolve().parents[2];OUTPUT=ROOT/'results/research/E020/contract_check.json'
def record(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
def main():
    assert not OUTPUT.exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5'
    started=time.time();r=dict(status='running',scope='Small GPU numerical/gradient/native contract; no model quality evidence')
    try:
        torch.manual_seed(20261003);torch.backends.cuda.matmul.allow_tf32=False
        r['device']=torch.cuda.get_device_name();assert torch.cuda.get_device_capability()==(12,0)
        x=torch.randn(32,64,device='cuda',dtype=torch.bfloat16)
        with collect_fastpack_checks():
            a=q.pack_nvfp4(x);b=pack_legacy_wan(x)
            assert torch.equal(a.packed,b.codes)
            assert torch.equal(a.swizzled_scales.view(torch.uint8),b.scales.view(torch.uint8))
            assert torch.equal(a.global_scale,b.global_scale)
            decoded=q.decode_packed(a);reference=a.decode();assert torch.equal(decoded,reference)
            r['bf16_packet_exact_existing_packer']=True;r['single_kernel_decode_equals_lut']=True
            for dtype in [torch.bfloat16,torch.float32]:
                src=x.to(dtype).detach().requires_grad_(True);got=q.qdq_ste(src)
                assert torch.equal(got,q.decode_packed(q.pack_nvfp4(src.detach())))
                got.float().sum().backward();assert torch.equal(src.grad,torch.ones_like(src))
            r['ste_forward_actual_qdq_and_identity_gradient']=True
            w=torch.ones(128,64,device='cuda',dtype=torch.float32);w[0,0]=1.001
            full=q.pack_nvfp4(w);rounded=q.pack_nvfp4(w.bfloat16())
            assert not torch.equal(full.global_scale,rounded.global_scale)
            r['fp32_master_not_prerounded_to_bf16']=dict(fp32_global=float(full.global_scale),bf16_global=float(rounded.global_scale))
            original=torch.nn.Linear(64,128,bias=True,device='cuda',dtype=torch.bfloat16)
            train=q.QADLinear(original);before=train.weight_master.detach().clone();output=train(x)
            output.float().square().mean().backward()
            assert train.weight_master.grad is not None and torch.isfinite(train.weight_master.grad).all() and train.weight_master.grad.abs().sum()>0
            opt=torch.optim.SGD([train.weight_master],lr=1e-3);opt.step()
            assert not torch.equal(before,train.weight_master)
            r['main_weight_gradient_and_update']=True
            packet=q.pack_nvfp4(train.weight_master.detach());native=q.PlainPackedWanLinear(packet,train.bias)
            real_mm=F.scaled_mm;actual=[]
            def counted(*args,**kwargs):
                actual.append(dict(a_dtype=str(args[0].dtype),b_dtype=str(args[1].dtype),shape_a=list(args[0].shape),shape_b=list(args[1].shape)))
                return real_mm(*args,**kwargs)
            F.scaled_mm=counted
            try:
                with torch.no_grad():actual_output=native(x)
            finally:F.scaled_mm=real_mm
            assert native.native_calls==1 and len(actual)==1
            assert actual[0]['a_dtype']==actual[0]['b_dtype']=='torch.float4_e2m1fn_x2'
            assert not list(native.parameters()) and not native._forward_hooks and not native._forward_pre_hooks
            with torch.no_grad():qdq_output=F.linear(q.decode_packed(q.pack_nvfp4(x)),q.decode_packed(packet),train.bias)
            error=(actual_output.float()-qdq_output.float()).double().square().sum();ref=qdq_output.double().square().sum()
            assert torch.isfinite(actual_output).all()
            r['actual_native_gemm']=actual;r['native_vs_qdq_nmse']=float(error/ref)
            r['native_has_no_trainable_or_bf16_weight']=True;r['packet_global']=record(packet.global_scale)
        torch.cuda.synchronize();r['status']='complete'
    except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        r.update(seconds_total=time.time()-started,sources={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),str(Path(q.__file__).resolve()):hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()})
        OUTPUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
