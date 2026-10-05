#!/usr/bin/env python3
"""Two fixed-packet BF16 calls diagnose E019's failed exact replay; no DiT.

False then True changes only return_lse. Outputs are persisted before any
numeric comparison. This does not relax or resume the original E019 gate.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import probe_h3_partition_contract as old
import torch

base, ROOT, RD = old.base, old.ROOT, old.RD
PLAN = ROOT/'research_state/06_experiments/E019_lse_diagnostic_amendment.md'
DATA = old.DATA/'lse_diagnostic'
require = old.require
KEYS = ('q_fp4','k_fp4','v_fp4_t','q_scale','k_scale','v_scale_t','correction')


def inputs(report):
    failed = json.loads((RD/'probe_run.json').read_text())
    require(failed['status']=='failed_stop' and failed['attention_calls']==1
            and failed['complete_dit_calls']==0, 'Original one-call failure changed')
    base.check_sources(failed['sources'])
    probe = json.loads(base.verify_file(failed['inherited']['probe_run.json']).read_text())
    case = next(r for r in probe['cases'] if r['block']==0)
    q, kv, phase0 = old.load_packets(case)
    row = failed['cases'][0]
    require(row['block']==0 and row['correction_e018_full_sha_exact'], 'Missing verified block0 correction')
    correction = torch.load(base.verify_file(row['correction']),map_location='cpu',weights_only=True,mmap=True)
    reference = torch.load(base.verify_file(phase0['output']['artifact']),map_location='cpu',weights_only=True,mmap=True)
    require(base.tensor_record(correction)==row['correction_tensor']==phase0['actual_q_packets']['correction'],
            'Saved GPU correction SHA changed')
    require(base.tensor_record(reference)==phase0['output']['tensor'], 'E018 reference tensor changed')
    require(correction.shape==(1,56,1,old.NP) and correction.dtype==torch.float32
            and reference.shape==(old.N,56,128) and reference.dtype==torch.bfloat16, 'Wrong correction/reference geometry')
    packet = dict(**q,**kv,correction=correction)
    finite = {k: bool(torch.isfinite(t.float()).all()) for k,t in packet.items() if t.is_floating_point()}
    require(all(finite.values()) and bool(torch.isfinite(reference).all()), 'Nonfinite saved input/reference')
    manifest = json.loads((ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    require(sys.executable==manifest['python'], 'Unexpected Python environment')
    for name,value in manifest['environment'].items():
        require(os.environ.get(name)==value, f'Native environment changed: {name}')
    environment = dict(python=sys.executable,torch=torch.__version__,torch_cuda=torch.version.cuda,
        packages={name:importlib.metadata.version(name) for name in ('torch','triton','flashinfer-python')})
    require(environment==failed['environment'], 'Package environment changed')
    records = dict(q_packets=phase0['q_packet_artifact'],kv_packets=case['kv_packets'],
                   correction=row['correction'],reference=phase0['output']['artifact'])
    report.update(environment=environment,sources={**failed['sources'],str(Path(__file__).resolve()):base.file_record(__file__),
        str(PLAN):base.file_record(PLAN)},
        prior_reports={name:base.file_record(RD/name) for name in ('probe_run.json','probe_check.json','frozen_contract.json')},
        e018_probe=failed['inherited']['probe_run.json'],input_files=records,input_tensors=base.tree_signature(packet),
        input_finite=finite,reference_tensor=base.tensor_record(reference),block=0,valid_length=old.N,padded_length=old.NP,
        call_order=[False,True],original_e019_gate='failed_stop, unchanged',original_attention_calls=1)
    return packet,reference


def difference(value, reference):
    """Independent FP64 chunked differences, including byte-only zero signs."""
    require(value.shape==reference.shape and value.dtype==reference.dtype, 'Comparison geometry changed')
    count,changed,changed_bytes,error2,ref2,maximum,delta_sum = value.numel(),0,0,0.,0.,0.,0.
    finite = True
    for begin in range(0,value.shape[0],256):
        a,b=value[begin:begin+256].contiguous(),reference[begin:begin+256].contiguous()
        changed += int((a!=b).sum())
        changed_bytes += int((a.view(torch.uint8)!=b.view(torch.uint8)).sum())
        x,y=a.double(),b.double()
        finite = finite and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        if finite:
            d=x-y
            error2+=float(d.square().sum()); ref2+=float(y.square().sum())
            maximum=max(maximum,float(d.abs().max())); delta_sum+=float(d.sum())
    return dict(elements=count,changed_elements=changed,changed_bytes=changed_bytes,
        bitwise_equal=changed_bytes==0,numerically_equal=changed==0,finite=finite,
        error_energy=error2 if finite else None,reference_energy=ref2 if finite else None,
        delta_mean=delta_sum/count if finite else None,rms=(error2/count)**.5 if finite else None,
        max_abs=maximum if finite else None,nmse=error2/ref2 if finite and ref2>0 else None)


def budget(args,report,before_call=False):
    require(time.time()<args.deadline_unix, 'Diagnostic absolute deadline expired')
    if before_call: require(report['attention_calls']<2, 'Two-call diagnostic budget exhausted')
    if torch.cuda.is_initialized():
        require(torch.cuda.max_memory_allocated()<=60*1024**3, '60-GiB peak allocation exceeded')


@torch.inference_mode()
def run(args,report,packet,reference):
    checked=old.complete(RD/'lse_diagnostic_check.json')
    for key in ('sources','prior_reports','e018_probe','input_files','input_tensors','reference_tensor','environment'):
        require(checked[key]==report[key], f'CPU precheck binding changed: {key}')
    require(checked['cuda_initialized'] is False, 'Missing CPU-only precheck')
    report['cpu_check']=base.file_record(RD/'lse_diagnostic_check.json')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='5', 'Only physical GPU5 authorized')
    budget(args,report)
    idle=subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True).strip()
    require([int(x.strip()) for x in idle.split(',')]==[0,0], f'GPU5 not idle: {idle}')
    report['gpu_before']=idle
    require(torch.cuda.get_device_capability()==(12,0), 'Expected SM120')
    torch.backends.cuda.matmul.allow_tf32=False
    import flashinfer.nvfp4_attention_sm120 as official
    DATA.mkdir(parents=True,exist_ok=False)
    gpu={k:packet[k].to('cuda') for k in KEYS}
    require(base.tree_signature(gpu)==report['input_tensors'], 'GPU upload changed input bytes')
    report['calls']=[]
    for flag in (False,True):
        budget(args,report,before_call=True)
        kwargs=dict(sm_scale=old.SCALE,causal=False,per_block_mean=False,out=None,lse=None,
                    out_dtype=torch.bfloat16,softmax_scale=None,return_lse=flag,unpadded_k_len=old.N)
        row=dict(return_lse=flag,kernel_kwargs={k:str(v) if k=='out_dtype' else v for k,v in kwargs.items()},
                 input_tensors=report['input_tensors'],status='calling')
        report['calls'].append(row); report['attention_calls']+=1
        base.save(report,args.output)
        result=official.nvfp4_attention_sm120_fwd(*(gpu[k] for k in KEYS),**kwargs)
        output,lse=result if flag else (result,None)
        torch.cuda.synchronize()
        full=output.cpu()
        valid=full[0,:,:old.N,:].transpose(0,1).contiguous()
        output_path=DATA/f'return_lse_{str(flag).lower()}_valid_output.pt'
        torch.save(valid,output_path)
        row['output']=base.file_record(output_path)
        if lse is not None:
            lse_cpu=lse.cpu()
            lse_path=DATA/'return_lse_true_full_lse.pt'
            torch.save(lse_cpu,lse_path)
            row['lse']=base.file_record(lse_path)
            row['lse_tensor']=base.tensor_record(lse_cpu)
            row['lse_finite']=bool(torch.isfinite(lse_cpu).all())
        row.update(status='saved',full_output_tensor=base.tensor_record(full),valid_output_tensor=base.tensor_record(valid))
        base.save(report,args.output)
        del output,lse,result,full,valid
        budget(args,report)
    # Both outputs are on disk before any output-equality gate or numerical conclusion.
    require(base.tree_signature(gpu)==report['input_tensors'], 'Kernel mutated original packet/correction')
    saved=[]
    for row in report['calls']:
        value=torch.load(base.verify_file(row['output']),map_location='cpu',weights_only=True,mmap=True)
        row.update(vs_e018=difference(value,reference),e018_sha_exact=row['valid_output_tensor']==report['reference_tensor'],status='complete')
        saved.append(value)
    report['true_vs_false']=difference(saved[1],saved[0])
    report['saved_before_comparisons']=True
    report['assessment']=('false_replay_failed_requires_further_localization' if not report['calls'][0]['e018_sha_exact'] else
        ('both_paths_replay_exact' if report['calls'][1]['e018_sha_exact'] else 'false_replays_exact_true_differs_no_causal_bug_claim'))
    require(report['attention_calls']==2, 'Incomplete diagnostic allocation')
    budget(args,report)
    report.update(status='complete',cuda_initialized=True,total_e019_attention_calls_including_failure=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    args.output=args.output or RD/f'lse_diagnostic_{args.phase}.json'
    require(not args.output.exists(),'Refusing to overwrite existing report')
    started=time.monotonic()
    torch.set_num_threads(6)
    report=dict(experiment='E019_lse_diagnostic',phase=args.phase,status='running',attention_calls=0,complete_dit_calls=0,
                scope='Fixed-packet return_lse diagnosis only; original E019 exact gate unchanged')
    try:
        if args.phase=='run':
            require(args.deadline_unix is not None, 'External absolute deadline is required')
            args.deadline_unix=min(args.deadline_unix,time.time()+300)
            report['deadline_unix']=args.deadline_unix
            budget(args,report)
        else:
            require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU precheck requires CUDA_VISIBLE_DEVICES empty')
        packet,reference=inputs(report)
        if args.phase=='check':
            require(not torch.cuda.is_initialized(), 'CPU precheck initialized CUDA')
            report.update(status='complete',cuda_initialized=False)
        else: run(args,report,packet,reference)
    except Exception:
        report.update(status='failed_stop',error=traceback.format_exc())
        raise
    finally:
        report['seconds_total']=time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        base.save(report,args.output)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__': main()
