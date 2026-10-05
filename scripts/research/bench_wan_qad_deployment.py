#!/usr/bin/env python3
"""E020 independent deployment timing: BF16 / SVD / plain / trained QAD.

Fresh models, full fixed input, two warmups and five synchronized-wall repeats.
No optimizer, training graph, profiler, generation, or cross-arm equality gate.
"""
from __future__ import annotations
import argparse,gc,json,os,statistics,time,traceback
from contextlib import contextmanager,nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from profile_wan_native_nvfp4 import BASE,RCM,CHECKPOINT,resident_model_storage
from train_wan_mainweight_qad import ROOT,RD,sha,save_json,tree
from bench_wan_native_nvfp4 import tensor_sha
import wan_mainweight_qad as qad
import wan_native_nvfp4 as native_lib
from wan_nvfp4_fastpack import pack_activation_fast,collect_fastpack_checks,validate_quantizer_contract

ARMS=('bf16','svd_nativefast','plain_step0000','qad_step0064')


@contextmanager
def counts():
    old_mm,old_sdpa=F.scaled_mm,F.scaled_dot_product_attention
    actual=dict(native_gemm=0,bf16_sdpa=0)
    def mm(a,b,*args,**kwargs):
        assert a.dtype==b.dtype==torch.float4_e2m1fn_x2
        actual['native_gemm']+=1
        return old_mm(a,b,*args,**kwargs)
    def sdpa(q,k,v,*args,**kwargs):
        assert q.dtype==k.dtype==v.dtype==torch.bfloat16
        actual['bf16_sdpa']+=1
        return old_sdpa(q,k,v,*args,**kwargs)
    F.scaled_mm,F.scaled_dot_product_attention=mm,sdpa
    try: yield actual
    finally: F.scaled_mm,F.scaled_dot_product_attention=old_mm,old_sdpa


def load_arm(arm,exports):
    from diffusers import WanPipeline,WanTransformer3DModel
    model=WanTransformer3DModel.from_pretrained(RCM,torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    info={}
    if arm=='svd_nativefast':
        from infer_rcm_wan_4step import load_quantized_transformer
        pipe=WanPipeline.from_pretrained(BASE,torch_dtype=torch.bfloat16,transformer=model,
            text_encoder=None,tokenizer=None,vae=None,local_files_only=True)
        load_quantized_transformer(pipe,CHECKPOINT,BASE)
        model=pipe.transformer.eval()
        conversion=native_lib.convert_wan_transformer_to_native(model,CHECKPOINT,activation_packer='legacy',chunk_rows=1024)
        assert conversion['target_count']==conversion['exact_roundtrip_count']==300
        modules=[m for m in model.modules() if isinstance(m,native_lib.NativeWanLinear)]
        assert len(modules)==300
        for module in modules:
            validate_quantizer_contract(module.activation_quantizer)
            module.activation_packer=pack_activation_fast
        info=dict(target_count=300,saved_weight_roundtrip_exact=300,checkpoint=str(CHECKPOINT))
        del pipe,modules,module,conversion
    elif arm!='bf16':
        row=exports[0 if arm=='plain_step0000' else 64]
        artifact=torch.load(row['file'],map_location='cpu',weights_only=True)
        modules=qad.install_packed(model,artifact)
        assert len(modules)==300
        info=dict(target_count=300,packed_file=row['file'],packed_sha256=row['sha256'],online_lowrank=False)
        del modules,artifact
    for p in model.parameters():p.requires_grad_(False)
    return model,info


def memory():
    return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


@torch.inference_mode()
def run_arm(arm,exports,call_args,call_kwargs,report,args,budget):
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();budget()
    row=dict(arm=arm,status='loading',before_load_memory=memory(),warmups=[],repeats=[])
    report['arms'].append(row);save_json(args.output,report)
    model,info=load_arm(arm,exports)
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();budget()
    row.update(model=info,resident_model_storage=resident_model_storage(model))
    torch.cuda.reset_peak_memory_stats()
    row['loaded_memory']=memory();expected_mm=0 if arm=='bf16' else 300
    first_sha=None
    try:
        for index in range(7):
            budget()
            checks_scope=nullcontext([]) if arm=='bf16' else collect_fastpack_checks()
            with counts() as actual,sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                with checks_scope as flags:
                    torch.cuda.synchronize();started=time.perf_counter()
                    result=model(*call_args,**call_kwargs)
                    y=result[0] if isinstance(result,(tuple,list)) else result.sample
                    torch.cuda.synchronize();finished=time.perf_counter()
                checked=time.perf_counter()
            # No tensor hashing, D2H output copy, or flag read is in the timer.
            assert actual==dict(native_gemm=expected_mm,bf16_sdpa=60) and len(flags)==expected_mm
            mem=memory();cpu=y.cpu();assert cpu.shape==(1,16,20,60,104) and torch.isfinite(cpu).all()
            digest=tensor_sha(cpu)
            if first_sha is None:first_sha=digest
            sample=dict(index=index if index<2 else index-2,wall_ms=(finished-started)*1000,
                flag_check_ms=(checked-finished)*1000,wall_including_checks_ms=(checked-started)*1000,
                counts=dict(actual),validated_fastpack_calls=len(flags),output_shape=list(cpu.shape),finite=True,
                output_sha256=digest,sha_equal_first_warmup=digest==first_sha,memory=mem)
            row['warmups' if index<2 else 'repeats'].append(sample)
            report['complete_dit_calls']+=1
            del result,y,cpu,flags,checks_scope
            budget();save_json(args.output,report)
        for key in ('wall_ms','wall_including_checks_ms','flag_check_ms'):
            values=[r[key] for r in row['repeats']]
            row[key]=dict(median=statistics.median(values),min=min(values),max=max(values))
        row.update(status='complete',memory_after_runs=memory())
    finally:
        del model
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        row['after_release_memory']=memory()
    save_json(args.output,report)
    print(arm,row['wall_ms']['median'],row['memory_after_runs'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-unix',type=float,required=True)
    parser.add_argument('--output',type=Path,default=RD/'deployment_bench.json')
    parser.add_argument('--train-report',type=Path,default=RD/'train_run.json')
    args=parser.parse_args();assert not args.output.exists()
    started=time.time();deadline=min(args.deadline_unix,started+1200)
    report=dict(experiment='E020_deployment',status='running',deadline_unix=deadline,arms=[],complete_dit_calls=0,
        protocol=dict(warmups=2,repeats=5,gpu=5,attention='BF16 FLASH_ATTENTION',max_seconds=1200,max_allocated_gib=60,
        timer='CUDA-synchronized host wall; excludes model loading, output D2H/hash, and flag reads; including-checks wall also reported',
        memory='Reset peaks after each model is loaded; includes warmups/repeats and resident shared input. No training process/optimizer.'))
    def budget():
        if time.time()>=deadline:raise TimeoutError('Deployment deadline expired')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>60*1024**3:raise MemoryError('60GiB allocation guard')
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5'
        train=json.loads(args.train_report.read_text());assert train['status']=='complete' and train['backward_steps']==64
        exports={r['step']:r for r in train['exports']};assert 0 in exports and 64 in exports
        for step in (0,64):assert sha(exports[step]['file'])==exports[step]['sha256']
        inventory_path=Path(train['settings']['inventory']);inventory=json.loads(inventory_path.read_text())
        selected=inventory['validation'][0];assert sha(selected['file'])==selected['sha256']
        payload=torch.load(selected['file'],map_location='cpu',weights_only=False)
        assert str(payload['filename'])==str(selected['prompt_id']) and payload['step']==selected['step']
        paths=[Path(__file__),Path(qad.__file__),Path(native_lib.__file__),ROOT/'scripts/research/wan_nvfp4_fastpack.py',
               ROOT/'scripts/research/profile_wan_native_nvfp4.py',ROOT/'scripts/infer_rcm_wan_4step.py',args.train_report,inventory_path]
        report['sources']={str(p.resolve()):sha(p) for p in paths}
        for p in (Path(qad.__file__),Path(native_lib.__file__),ROOT/'scripts/research/wan_nvfp4_fastpack.py'):
            assert sha(p)==train['sources'][str(p.resolve())],'Training/deployment source drift'
        assert payload['input_args'][0].shape==(1,16,20,60,104) and payload['input_kwargs']['encoder_hidden_states'].shape==(1,512,4096)
        torch.set_num_threads(6);torch.manual_seed(20261003);torch.backends.cuda.matmul.allow_tf32=False
        assert torch.cuda.get_device_capability()==(12,0)
        report.update(input=selected,environment=dict(python=os.sys.executable,torch=torch.__version__,device=torch.cuda.get_device_name()))
        call_args,call_kwargs=tree(payload['input_args'],'cuda'),tree(payload['input_kwargs'],'cuda')
        del payload
        for arm in ARMS:run_arm(arm,exports,call_args,call_kwargs,report,args,budget)
        assert report['complete_dit_calls']==28
        del call_args,call_kwargs;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        report.update(status='complete',final_memory=memory())
    except BaseException:report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds_total']=time.time()-started
        save_json(args.output,report);print(report['status'],flush=True)


if __name__=='__main__':main()
