"""Bounded E024 official environment check; no model load or source modification."""
import argparse, collections, hashlib, importlib, importlib.metadata, json, os, pathlib, subprocess, sys, time, traceback
ROOT=pathlib.Path('/home/wjq/workspace/svdquant-exp')
SRC=pathlib.Path('/data1/models/svdquant-wjq/third_party/FastVideo-8444c089')
REV='8444c0897a8b96848eb85b6e5750ef486f79fc92'

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','smoke'],required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args()
    assert not a.output.exists(),a.output
    r={'status':'running','phase':a.phase,'model_calls':0,'actual_calls':{},'imports':{},'started_unix':time.time(),'source_commit':subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    def save():a.output.write_text(json.dumps(r,indent=2)+'\n')
    def budget():
        if time.time()>=a.deadline_unix:raise TimeoutError('deadline')
    try:
        assert r['source_commit']==REV;budget()
        import torch
        r.update(python=sys.version,torch=torch.__version__,torch_cuda_build=torch.version.cuda,packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},cuda_initialized_before=torch.cuda.is_initialized())
        assert torch.__version__=='2.12.0+cu130' and torch.version.cuda=='13.0'
        if a.phase=='smoke':
            assert os.environ['CUDA_VISIBLE_DEVICES']=='1'; torch.cuda.set_device(0)
            props=torch.cuda.get_device_properties(0);torch.cuda.set_per_process_memory_fraction(10*2**30/props.total_memory,0)
            r['device']={'name':props.name,'capability':list(torch.cuda.get_device_capability()),'total_bytes':props.total_memory}
            assert tuple(r['device']['capability'])==(12,0)
        for name in ['fp4attn_cuda','fp4quant_cuda','attn_qat_infer','flashinfer','taehv','torchvision','torchaudio','fastvideo_kernel','fastvideo']:
            try:
                m=importlib.import_module(name); r['imports'][name]={'status':'pass','file':getattr(m,'__file__',None)}
            except Exception:
                s=traceback.format_exc();r['imports'][name]={'status':'failed','traceback':s}
                if a.phase=='smoke':raise
        if a.phase=='cpu':
            failed={k:v for k,v in r['imports'].items() if v['status']=='failed'}
            assert all(k in ['fastvideo','fastvideo_kernel'] and '0 active drivers' in v['traceback'] for k,v in failed.items()),failed
            r['status']='cpu_checked_gpu_import_pending' if failed else 'complete';assert not torch.cuda.is_initialized()
        else:
            from fastvideo.layers.quantization.nvfp4_qat_config import NVFP4QATQuantizeMethod,convert_model_to_fp4
            from fastvideo.attention.backends.attn_qat_infer import attn_qat_infer_receipt
            import flashinfer,fp4attn_cuda,attn_qat_infer
            r['backend']=attn_qat_infer_receipt();assert 'kernel=fastvideo-kernel-cutlass' in r['backend']
            calls=collections.Counter();saved=[]
            def watch(module,name,key):
                original=getattr(module,name);saved.append((module,name,original))
                def tracked(*args,**kwargs):calls[key]+=1;r['actual_calls']=dict(calls);return original(*args,**kwargs)
                setattr(module,name,tracked)
            watch(flashinfer,'mm_fp4','nvfp4_gemm');watch(flashinfer,'nvfp4_quantize','nvfp4_linear_quantize');watch(fp4attn_cuda,'fwd','fp4_attention')
            torch.manual_seed(20261003);torch.cuda.reset_peak_memory_stats();budget()
            layer=torch.nn.Module();layer.register_parameter('weight',torch.nn.Parameter(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*0.02,requires_grad=False));layer.quant_method=NVFP4QATQuantizeMethod();convert_model_to_fp4(layer)
            assert 'weight' not in layer._parameters
            x=torch.randn(1,256,128,device='cuda',dtype=torch.bfloat16)
            q,k,v=[torch.randn(1,2,256,128,device='cuda',dtype=torch.bfloat16)*0.2 for _ in range(3)]
            with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                y=layer.quant_method.apply(layer,x)
                z=attn_qat_infer.sageattn_blackwell(q,k,v,is_causal=False,per_block_mean=True,single_level_p_quant=True,sm_scale=128**-0.5)
                torch.cuda.synchronize()
            r['actual_calls']=dict(calls)
            r['outputs']={name:{'shape':list(t.shape),'dtype':str(t.dtype),'finite':bool(torch.isfinite(t).all().item()),'sha256':hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()} for name,t in [('linear',y),('attention',z)]}
            r['inputs']={'linear_x':[1,256,128],'weight':[128,128],'qkv_each':[1,2,256,128],'dtype':'torch.bfloat16','seed':20261003}
            r['cuda_kernels']=[{'name':e.name,'count':e.count,'device_us':e.device_time_total} for e in prof.key_averages() if str(e.device_type)=='DeviceType.CUDA']
            r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            for module,name,fn in saved:setattr(module,name,fn)
            assert calls=={'nvfp4_gemm':1,'nvfp4_linear_quantize':2,'fp4_attention':1},calls
            assert all(t['finite'] for t in r['outputs'].values())
            assert y.shape==x.shape and z.shape==q.shape and y.dtype==z.dtype==torch.bfloat16
            assert r['peak_reserved_bytes']<=10*2**30
            budget();r['status']='complete'
        r['cuda_initialized_after']=torch.cuda.is_initialized()
    except Exception:
        r['status']='failed';r['error']=traceback.format_exc();raise
    finally:r['finished_unix']=time.time();save()
if __name__=='__main__':main()
