#!/usr/bin/env python3
"""E009b: bounded FlashInfer SM120 fused-up contract and component timing.
Same packed residual operands; original BF16 smoothed-x down branch. This is a
known-kernel deployment check, not a new method or full-DiT speed/quality result.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research')]
from smoke_nvfp4_systems_20261002 import swizzle,operand


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def metric(y,r):
    e2=r2=ma=0.
    yf,rf=y.flatten(),r.flatten()
    for i in range(0,rf.numel(),1024**2):
        a,b=yf[i:i+1024**2].float(),rf[i:i+1024**2].float()
        d=a-b;e2+=d.square().sum(dtype=torch.float64).item()
        r2+=b.square().sum(dtype=torch.float64).item();ma=max(ma,d.abs().max().item())
    return dict(err2=e2,ref2=r2,nmse=e2/max(r2,1e-30),max_abs=ma)


def save(rep,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(rep,indent=2)+'\n');tmp.replace(p)


def torch_main(a,b,sa,sb,ga,gb):
    rec=[F.ScalingType.BlockWise1x16,F.ScalingType.TensorWise]
    return F.scaled_mm(a.view(torch.float4_e2m1fn_x2),b.view(torch.float4_e2m1fn_x2).T,
        [sa.view(torch.float8_e4m3fn),ga],rec,[sb.view(torch.float8_e4m3fn),gb],rec,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,output_dtype=torch.bfloat16)


def timed(fn,repeats=5):
    fn();torch.cuda.synchronize();rows=[]
    for _ in range(repeats):
        st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();t=time.perf_counter();st.record();y=fn();en.record();en.synchronize()
        rows.append({'wall_ms':(time.perf_counter()-t)*1000,'device_ms':st.elapsed_time(en)})
        if not bool(y.isfinite().all()):raise RuntimeError('timed output nonfinite')
    return {'repeats':rows,'median_wall_ms':statistics.median(r['wall_ms'] for r in rows),
            'median_device_ms':statistics.median(r['device_ms'] for r in rows)}


@torch.inference_mode()
def execute(args,rep):
    torch.set_num_threads(6);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False
    from flashinfer.gemm import mm_nvfp4_svdquant
    import flashinfer.gemm.gemm_svdquant as module
    import cutlass
    import cutlass.cute
    rep['versions']={n:metadata.version(n) for n in ['torch','flashinfer-python','nvidia-cutlass-dsl','nvidia-cutlass-dsl-libs-base','cuda-python','cuda-bindings','apache-tvm-ffi']}
    rep.update(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
               torch_cuda=torch.version.cuda,environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_HOME','CUTE_DSL_CUDA_VERSION','FLASHINFER_WORKSPACE_BASE','TRITON_CACHE_DIR']})
    rep['files']={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [Path(__file__),Path(module.__file__),ROOT/'scripts/research/smoke_nvfp4_systems_20261002.py']}
    save(rep,args.output)
    # Finite orthogonal tests catch both residual and up scaling/layout errors.
    for m,n,k in [(128,128,256),(257,256,512)]:
        a,sa,ar=operand(m,k);b,sb,br=operand(n,k)
        a=a.view(torch.uint8);b=b.view(torch.uint8);sa=sa.view(torch.uint8);sb=sb.view(torch.uint8)
        ga=torch.tensor([0.0137],device='cuda');gb=torch.tensor([0.00231],device='cuda');alpha=ga*gb
        d=torch.randn(m,32,device='cuda',dtype=torch.bfloat16)*0.02
        up=torch.randn(n,32,device='cuda',dtype=torch.bfloat16)*0.02
        l1=(up.float()/alpha).to(torch.bfloat16).contiguous()
        for label,aa,dd in [('main_only',a,torch.zeros_like(d)),('up_only',torch.zeros_like(a),d),('combined',a,d)]:
            print(f'synthetic {m} {n} {k} {label}',flush=True)
            ref=((ar@br.T) if label!='up_only' else torch.zeros(m,n,device='cuda'))
            ref=(ref+dd.float()@l1.float().T)*alpha
            t=time.perf_counter()
            y=mm_nvfp4_svdquant(aa,b,sa,sb,alpha,dd,l1,backend='cute-dsl',enable_pdl=False)
            torch.cuda.synchronize();startup=time.perf_counter()-t
            u=mm_nvfp4_svdquant(aa,b,sa,sb,alpha,dd,l1,backend='cute-dsl-unfused',enable_pdl=False)
            row={'shape_mnk':[m,n,k],'arm':label,'fused_vs_fp32_same_packet_and_scaled_l1':metric(y,ref),
                'unfused_vs_fp32_same_packet_and_scaled_l1':metric(u,ref),'fused_vs_flashinfer_unfused':metric(y,u),
                'first_call_seconds_not_benchmark':startup,'finite':bool(y.isfinite().all() and u.isfinite().all())}
            if label=='combined':
                original=torch_main(a,b,sa,sb,ga,gb)+F.linear(d,up)
                row['fused_vs_original_torch_main_and_bf16_up']=metric(y,original)
                row['l1_rescale_reconstruction_vs_original_up']=metric((l1.float()*alpha).to(up),up)
            if not row['finite'] or row['fused_vs_fp32_same_packet_and_scaled_l1']['nmse']>1e-4:
                raise RuntimeError(f'synthetic scaling gate failed: {row}')
            rep['synthetic'].append(row);save(rep,args.output)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            y=mm_nvfp4_svdquant(a,b,sa,sb,alpha,d,l1,backend='cute-dsl',enable_pdl=False);torch.cuda.synchronize()
        kernels=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})
        rep.setdefault('fused_kernel_names',[]).append({'shape_mnk':[m,n,k],'names':kernels})
        if not kernels:raise RuntimeError('no actual fused GPU kernels observed')
        save(rep,args.output)
    if args.inputs:
        run_real(args,rep,mm_nvfp4_svdquant)
    rep['status']='complete';save(rep,args.output)


@torch.inference_mode()
def run_real(args,rep,mm):
    from h3_native_nvfp4 import pack_activation_legacy
    from wan_native_nvfp4 import PackedNVFP4
    if args.export_report is None:
        raise ValueError('real inputs require exporter smoke report')
    source_raw=args.export_report.read_bytes()
    source=json.loads(source_raw)
    snapshot=args.output.with_name(args.output.stem+'_export_snapshot.json')
    snapshot.write_bytes(source_raw)
    if len(source.get('smoke',[]))!=2:
        raise ValueError('two real-layer exporter gates have not both completed')
    lookup={r['fused_input_artifact']['path']:r for r in source['smoke']}
    rep['export_report']={'path':str(args.export_report),'sha256':sha(args.export_report),
                          'status_at_read':source['status'],'snapshot_path':str(snapshot),
                          'snapshot_sha256':sha(snapshot),'two_layer_smoke':source['smoke']}
    for file in [ROOT/'scripts/research/h3_native_nvfp4.py',ROOT/'scripts/research/wan_native_nvfp4.py']:
        rep['files'][str(file)]={'sha256':sha(file),'bytes':file.stat().st_size}
    for path in args.inputs:
        gate=lookup[str(path)]
        if sha(path)!=gate['fused_input_artifact']['sha256']:
            raise RuntimeError('real-input artifact hash changed')
        for name in ['activation_roundtrip','bypass_vs_original_hooks','packed_qdq_vs_original_hooks']:
            if not gate[name]['exact']:
                raise RuntimeError('exporter numerical gate did not pass')
        item=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if item['format']!='E009_h3_fused_input_v1':raise ValueError('unexpected real-input schema')
        x_all=item['x_s'].cuda();a_lr=item['a'].cuda();b_lr=item['b'].cuda()
        w=item['weight_packed'].cuda();sw=item['weight_scales_swizzled'].cuda()
        gw=item['weight_global'].cuda().reshape(1)
        if x_all.ndim!=2 or x_all.shape[0]!=22400:raise ValueError('expected full22400 H3 tokens')
        for rows in [512,22400]:
            print(f"real {item['layer']} M={rows}",flush=True)
            x=x_all[:rows].contiguous();packet=pack_activation_legacy(x)
            ac,sa,ga=packet.packed,packet.swizzled_scales,packet.global_scale
            alpha=ga*gw;down=F.linear(x,a_lr)
            # Explicit FP32 arithmetic avoids a half-precision alpha roundtrip.
            l1=(b_lr.float()/alpha).to(torch.bfloat16).contiguous()
            if not bool(l1.isfinite().all()):raise RuntimeError('scaled BF16 LoRA-up overflow')
            api=lambda d,l,backend='cute-dsl':mm(ac,w,sa.view(torch.uint8),sw.view(torch.uint8),alpha,d,l,
                backend=backend,enable_pdl=False)
            original_main=torch_main(ac,w,sa,sw,ga,gw)
            fused_main=api(torch.zeros_like(down),l1)
            main_check=metric(fused_main,original_main)
            if not bool(fused_main.isfinite().all()) or main_check['nmse']>1e-4:
                raise RuntimeError(f'real samepacket main gate failed: {main_check}')
            del fused_main
            original_up=F.linear(down,b_lr)
            original=original_main+original_up
            # The actual fused numerical contract uses BF16 l1=B/alpha then an
            # FP32 accumulation epilogue. Record rescale and final-round effects.
            l1_error=metric((l1.float()*alpha).to(b_lr),b_lr)
            scaled_up=F.linear(down,l1)
            scaled_up.mul_(alpha)
            up_error=metric(scaled_up,original_up)
            del scaled_up,original_up,original_main
            fused=api(down,l1);unfused=api(down,l1,'cute-dsl-unfused')
            row={'layer':item['layer'],'input_path':str(path),'input_sha256':sha(path),
                 'shape_mnk':[rows,w.shape[0],w.shape[1]*2],'alpha':float(alpha),
                 'samepacket_fused_main_vs_torch_native':main_check,
                 'scaled_l1_reconstruction_vs_original_b':l1_error,
                 'rescaled_bf16_up_vs_original_bf16_up':up_error,
                 'fused_vs_original_torch_native_plus_bf16_up':metric(fused,original),
                 'flashinfer_unfused_vs_original_torch_native_plus_bf16_up':metric(unfused,original),
                 'fused_vs_flashinfer_unfused':metric(fused,unfused),
                 'finite':bool(fused.isfinite().all() and unfused.isfinite().all())}
            if not row['finite']:raise RuntimeError('real combined output nonfinite')
            del original,fused,unfused
            # Only prepared-packet components are timed. No pack/smooth cost is
            # hidden in a claim of full linear latency.
            core={
                'torch_native_main_plus_original_up':lambda:torch_main(ac,w,sa,sw,ga,gw)+F.linear(down,b_lr),
                'flashinfer_unfused_prepared':lambda:api(down,l1,'cute-dsl-unfused'),
                'flashinfer_fused_prepared':lambda:api(down,l1),
            }
            def torch_chain():
                d=F.linear(x,a_lr)
                return torch_main(ac,w,sa,sw,ga,gw)+F.linear(d,b_lr)
            def fused_chain():
                # Globals are device tensors; include alpha and l1 arithmetic.
                dynamic_alpha=ga*gw
                dynamic_l1=(b_lr.float()/dynamic_alpha).to(torch.bfloat16)
                d=F.linear(x,a_lr)
                return mm(ac,w,sa.view(torch.uint8),sw.view(torch.uint8),dynamic_alpha,d,dynamic_l1,
                    backend='cute-dsl',enable_pdl=False)
            row['timings_prepared_packet_only']={name:timed(fn) for name,fn in core.items()}
            row['timings_with_down_and_dynamic_rescale_excludes_pack_smooth']={
                'torch':timed(torch_chain),'flashinfer_fused':timed(fused_chain)}
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                y=api(down,l1);torch.cuda.synchronize()
            row['actual_fused_kernels']=sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})
            del y
            rep['real'].append(row);save(rep,args.output)
            del packet,down,l1,core
        del item,x_all,a_lr,b_lr,w,sw,gw
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,nargs='*')
    p.add_argument('--export-report',type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rep={'status':'running','synthetic':[],'real':[],'scope':'known fused-up contract; no full linear/DiT speed or quality claim'}
    t=time.perf_counter()
    try:execute(args,rep)
    except Exception:
        rep.update(status='failed',traceback=traceback.format_exc());raise
    finally:
        rep['elapsed_seconds']=time.perf_counter()-t
        if torch.cuda.is_initialized():rep['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        save(rep,args.output)

if __name__=='__main__':main()
