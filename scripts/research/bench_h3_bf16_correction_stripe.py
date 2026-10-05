#!/usr/bin/env python3
"""E027 isolated BF16 M16 correction-stripe cost; not a fused-attention bound.

One CTA owns one (head, query group) and traverses all 177 key tiles in reverse.
Every dot is M16/N128/K128: one useful mean row, fifteen repeated unused rows.
This has its own cache behavior, launch, occupancy and checksum stores. Its time
cannot be subtracted from materialized-attention time to predict a fusion gain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E027'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E027/stripe')
CAPTURE=ROOT/'results/research/E018/capture_run.json'
H,G,NT,D,N,NP=56,177,177,128,22539,22656


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()


def record(path):
    p=Path(path).absolute();return dict(file=str(p),bytes=p.stat().st_size,sha256=sha(p))


def write(path,r):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def inputs(report):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    old=json.loads(CAPTURE.read_text());assert old['status']=='complete'
    row=next(c for c in old['cases'] if c['block']==0)
    assert record(row['artifact']['file'])==row['artifact']
    raw=torch.load(row['artifact']['file'],map_location='cpu',weights_only=True,mmap=True)
    for key in ('q','k','v'):assert raw[key].shape==(N,H,D) and raw[key].dtype==torch.bfloat16
    report.update(source=record(__file__),capture_report=record(CAPTURE),input=row['artifact'],
        official_source=record(official.__file__),source_case=raw['source_case'])
    return raw


def build_kernel():
    import triton
    import triton.language as tl
    @triton.jit
    def stripe(MU,K,OUT,NVALID:tl.constexpr,NPAD:tl.constexpr,NGROUP:tl.constexpr,NTILES:tl.constexpr):
        hg=tl.program_id(0);h=hg//NGROUP
        r=tl.arange(0,16);d=tl.arange(0,128);j=tl.arange(0,128)
        mu=tl.load(MU+hg*128+d)
        # Repeated rows are explicitly unused. The executed MMA has M16.
        a=tl.broadcast_to(mu[None,:],(16,128))
        for step in range(NTILES):
            tile=NTILES-1-step;n=tile*128+j
            b=tl.load(K+h*NPAD*128+n[None,:]*128+d[:,None],mask=n[None,:]<NVALID,other=0)
            acc=tl.dot(a,b,out_dtype=tl.float32)
            firstrow=tl.sum(tl.where(r[:,None]==0,acc,0.),axis=0)
            last=tl.minimum(127,NVALID-1-tile*128)
            first_value=tl.sum(tl.where(j==0,firstrow,0.),axis=0)
            last_value=tl.sum(tl.where(j==last,firstrow,0.),axis=0)
            checksum=tl.sum(firstrow,axis=0)
            ptr=OUT+(hg*NTILES+tile)*3
            tl.store(ptr,first_value);tl.store(ptr+1,last_value);tl.store(ptr+2,checksum)
    return stripe


def comparison(a,b):
    a=a.double();b=b.double();e=a-b
    err=float(e.square().sum());energy=float(b.square().sum())
    return dict(nmse=err/energy if energy else None,rms=(err/e.numel())**.5,max_abs=float(e.abs().max()),
                reference_energy=energy,error_energy=err,elements=e.numel())


def run(args,report,raw):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert args.deadline_unix and 0<args.deadline_unix-time.time()<=600
    assert torch.cuda.get_device_capability()==(12,0)
    def budget():
        if time.time()>=args.deadline_unix:raise TimeoutError('E027 stripe external deadline')
    assert not DATA.exists();DATA.mkdir(parents=True)
    # Set before the first Triton import/compilation; no shared or root JIT cache writes.
    os.environ['TRITON_CACHE_DIR']=str(DATA/'triton_cache')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,20*2**30/torch.cuda.get_device_properties(0).total_memory))
    segment=tuple(raw[k].transpose(0,1).unsqueeze(0).contiguous().cuda() for k in ('q','k','v'))
    with torch.inference_mode():
        budget()
        processed=official._preprocess_qkv(*segment,per_block_mean=True)
        k=processed[1][0].contiguous();reference=processed[3][0]
        mu=official._pad_seq_len_to_128(segment[0]).reshape(1,H,G,128,D).mean(3)[0].contiguous()
        assert k.dtype==mu.dtype==torch.bfloat16 and reference.dtype==torch.float32
        # Exact original BF16 operands; official FP32 matmul supplies the reference.
        assert bool(mu.isfinite().all()) and bool(k.isfinite().all()) and bool(reference.isfinite().all())
        del processed,segment
        out=torch.empty((H,G,NT,3),device='cuda',dtype=torch.float32)
        kernel=build_kernel();compiled=None
        for _ in range(3):
            budget();compiled=kernel[(H*G,)](mu,k,out,N,NP,G,NT,num_warps=4,num_stages=1)
            report['stripe_launches']+=1
        torch.cuda.synchronize()
        ptx=compiled.asm['ptx'];mma=[line.strip() for line in ptx.splitlines() if 'mma.sync' in line and 'm16n8k16' in line and '.bf16.bf16.' in line]
        ptx_path=DATA/'stripe.ptx';ptx_path.write_text(ptx)
        report['ptx']=record(ptx_path);report['bf16_mma_instructions']=mma
        assert mma,'Actual compiler output did not contain expected BF16 tensor-core MMA'
        ms=[]
        for _ in range(10):
            budget();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();kernel[(H*G,)](mu,k,out,N,NP,G,NT,num_warps=4,num_stages=1);end.record()
            report['stripe_launches']+=1;end.synchronize();ms.append(start.elapsed_time(end))
        report['timing_ms']=dict(repeats=ms,median=statistics.median(ms),min=min(ms),max=max(ms),warmup=3)
        assert bool(out.isfinite().all())
        ref=reference.reshape(H,G,NT,128)
        expected=torch.stack((ref[:,:,:,0],ref[:,:,:,127].clone(),ref.sum(-1)),dim=-1)
        expected[:,:,-1,1]=ref[:,:,-1,N-(NT-1)*128-1]
        values=out.cpu();expected=expected.cpu()
        report['numeric_check']={key:comparison(values[...,i],expected[...,i])
                                 for i,key in enumerate(('tile_first','tile_last_valid','tile_sum'))}
        heads=[0,8,16,24,32,40,48,55];groups=[0,88,176];tiles=[0,88,176]
        sample=[]
        for h in heads:
            for g in groups:
                for t in tiles:sample.append(dict(head=h,query_group=g,key_tile=t,actual=values[h,g,t].tolist(),reference=expected[h,g,t].tolist()))
        report['samples']=sample;report['output_finite']=True
        report['compiled_resources']=dict(shared_bytes=compiled.metadata.shared,num_warps=compiled.metadata.num_warps,
            registers=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None))
        report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        report['environment']=dict(torch=torch.__version__,device=torch.cuda.get_device_name(),triton_cache=os.environ['TRITON_CACHE_DIR'])
        budget();assert report['stripe_launches']==13
        report.update(status='complete',cuda_initialized=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('check','run'),required=True)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--output',type=Path)
    a=p.parse_args();a.output=a.output or RD/('stripe_'+a.phase+'.json');assert not a.output.exists()
    start=time.time();report=dict(experiment='E027',component='bf16_correction_stripe',status='running',phase=a.phase,
        stripe_launches=0,actual_dit_calls=0,actual_attention_calls=0,
        geometry=dict(heads=H,query_groups=G,key_tiles=NT,valid_tokens=N,padded_tokens=NP,m=16,n=128,k=128,
                      useful_mean_rows=1,repeated_unused_rows=15,ctas=H*G,tiles_per_launch=H*G*NT),
        scope='Isolated BF16 MMA+FP32 accumulation traversing full K. Three scalar checksum stores per tile. Not fused latency, not a lower bound, and not an attention or end-to-end speedup measurement.')
    try:
        import torch
        torch.set_num_threads(6)
        if a.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        raw=inputs(report)
        if a.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            checked=json.loads((RD/'stripe_check.json').read_text());assert checked['status']=='complete'
            assert checked['source']==report['source'] and checked['input']==report['input'] and checked['official_source']==report['official_source']
            run(a,report,raw)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-start;write(a.output,report)
        print(json.dumps(dict(status=report['status'],report=str(a.output))),flush=True)


if __name__=='__main__':main()
