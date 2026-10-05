#!/usr/bin/env python3
"""E039 full original attention O, including real model padding; two SDPA only."""
import argparse, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E039'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E039')
SOURCE=Path('/data1/models/svdquant-wjq/research/20261003/E035/capture/block_00.pt')
PLAN=ROOT/'research_state/06_experiments/E039_output_projection_parallel_plan.md'
def record(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return dict(file=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
def save(d,p):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def launch():
    from resume_h3_plain_baseline import idle_stable
    output=RD/'prepare_launcher.json';assert not output.exists();assert not (RD/'prepare.json').exists()
    gpu=idle_stable(0);start=time.time();deadline=start+300
    m=json.loads((ROOT/'research_state/06_experiments/E038_center_video_manifest.json').read_text());python=m['python']
    env=dict(os.environ,**m['environment'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env['PATH']=str(Path(python).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
    env.update(TRITON_CACHE_DIR='/data1/models/svdquant-wjq/research/cache/triton',CUDA_CACHE_PATH='/data1/models/svdquant-wjq/research/cache/cuda')
    log=ROOT/'results/logs/E039_prepare.log';command=[python,'-u',str(Path(__file__).resolve()),'--run','--deadline-unix',str(deadline)]
    report=dict(experiment='E039',status='running',gpu_before=gpu,command=command,deadline_epoch=deadline,log=str(log));proc=None
    try:
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report['pid']=proc.pid;save(report,output);report['returncode']=proc.wait(timeout=deadline-time.time())
        assert report['returncode']==0
        result=json.loads((RD/'prepare.json').read_text());assert result['status']=='complete' and result['actual_sdpa_calls']==2
        report['status']='complete'
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        report['seconds']=time.time()-start;save(report,output);print(json.dumps(report),flush=True)
def run(deadline):
    assert deadline and time.time()<deadline and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    output=RD/'prepare.json';assert not output.exists();assert not (DATA/'attention_output.pt').exists()
    import run_h3_native_paired_video as old
    import torch
    from diffsynth.models import minimax_h3_dit as model
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.time();report=dict(experiment='E039',status='running',actual_sdpa_calls=0,actual_dit_calls=0,receipts=[])
    original=torch.nn.functional.scaled_dot_product_attention
    def counted(q,k,v,*args,**kwargs):
        assert time.time()<deadline;report['actual_sdpa_calls']+=1;assert report['actual_sdpa_calls']<=2
        report['receipts'].append(dict(q_shape=list(q.shape),dtype=str(q.dtype),scale=kwargs.get('scale')))
        return original(q,k,v,*args,**kwargs)
    try:
        payload=torch.load(SOURCE,map_location='cpu',weights_only=True,mmap=True)
        assert payload['block']==0 and payload['valid_length']==22539 and payload['total_tokens']==22592
        cu=payload['kwargs']['cu_seqlens'];assert cu.tolist()==[0,22539,22592]
        for name in ('q','k','v'):
            x=payload['native'][name];assert x.shape==(22592,56,128) and x.dtype==torch.bfloat16 and x.device.type=='cpu'
        report.update(source=record(SOURCE),plan=record(PLAN),sources=[record(__file__),record(model.__file__)],
            torch=torch.__version__,cuda=torch.version.cuda,input_qkv_kind='E035 native projection, same original BF16 teacher input state',
            valid_length=22539,model_total=22592,model_padding=53,cu_seqlens=cu.tolist(),scale=payload['scale'])
        q,k,v=(payload['native'][name].cuda() for name in ('q','k','v'))
        torch.nn.functional.scaled_dot_product_attention=counted
        with torch.inference_mode():result=model._sdpa_varlen_attention(q,k,v,cu,payload['scale'])
        torch.cuda.synchronize();assert report['actual_sdpa_calls']==2 and result.shape==(22592,56,128)
        assert result.dtype==torch.bfloat16 and bool(torch.isfinite(result).all())
        DATA.mkdir(parents=True,exist_ok=True);path=DATA/'attention_output.pt';torch.save(result.cpu(),path)
        report.update(status='complete',artifact=record(path),shape=list(result.shape),dtype=str(result.dtype),finite=True,
            model_padding_rms=float(result[22539:].float().square().mean().sqrt()),
            model_padding_max_abs=float(result[22539:].abs().max()),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        assert time.time()<deadline
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        torch.nn.functional.scaled_dot_product_attention=original;report['seconds']=time.time()-start
        save(report,output);print(json.dumps(dict(status=report['status'],output=str(output))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.run:run(a.deadline_unix)
    else:launch()
