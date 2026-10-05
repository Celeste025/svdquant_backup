#!/usr/bin/env python3
"""E043: bound four independent official Wan teacher workers; no retries."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'research_state/06_experiments/E043_vanilla_wan_manifest.json'
RUNNER=ROOT/'scripts/research/run_vanilla_wan_reference.py'
RD=ROOT/'results/research/E043'
LOGS=ROOT/'results/logs'


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def record(path):
    return dict(file=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def devices():
    cmd=['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    raw=subprocess.check_output(cmd,text=True)
    return {int(r[0]):dict(memory_used_mib=int(r[1]),utilization_pct=int(r[2])) for r in
            ([x.strip() for x in line.split(',')] for line in raw.strip().splitlines())}


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    manifest=json.loads(MANIFEST.read_text())
    RD.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    output=RD/'launcher.json'
    assert not output.exists(), 'Keep prior attempts'
    assert RUNNER.is_file()
    started=time.time();deadline=started+manifest['budget']['launcher_seconds']
    report=dict(experiment='E043',status='running',start_epoch=started,deadline_epoch=deadline,
                manifest=record(MANIFEST),runner=record(RUNNER),launcher=record(Path(__file__)),workers=[])
    owned=[];streams=[]
    try:
        observations=[]
        for sample in range(2):
            current=devices();observations.append(current)
            for gpu in manifest['budget']['gpus'].values():
                assert current[gpu]['memory_used_mib']<256 and current[gpu]['utilization_pct']==0, f'GPU {gpu} busy'
            if sample==0:time.sleep(2)
        report['gpu_before']=observations
        for pid,gpu in manifest['budget']['gpus'].items():
            result=RD/f'worker_{pid}.json';log=LOGS/f'E043_worker_{pid}.log'
            assert not result.exists() and not log.exists(), 'Keep existing worker artifacts'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_HOME='/usr/local/cuda',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                     OMP_NUM_THREADS='4',MAX_JOBS='4',PYTHONUNBUFFERED='1')
            cache=Path('/data1/models/svdquant-wjq/research/cache')
            for key,sub in {'HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg','TORCH_HOME':'torch',
                            'TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton',
                            'CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'torch_extensions','TMPDIR':'tmp'}.items():
                path=cache/sub;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
            env['FLASHINFER_WORKSPACE_BASE']=str(cache/'flashinfer')
            env['PATH']=str(Path(manifest['python']).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
            worker_deadline=min(deadline,time.time()+manifest['budget']['worker_seconds'])
            cmd=[manifest['python'],'-u',str(RUNNER),'--prompt-id',pid,'--output',str(result),
                 '--deadline-unix',str(worker_deadline)]
            stream=log.open('x');streams.append(stream)
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            owned.append((proc,worker_deadline,result))
            report['workers'].append(dict(prompt_id=int(pid),gpu=gpu,pid=proc.pid,deadline_epoch=worker_deadline,
                                          command=cmd,log=str(log),output=str(result),status='running'))
            save(output,report);print(f'Launched prompt {pid} on GPU{gpu}, worker {proc.pid}',flush=True)
        while True:
            all_done=True
            for i,(proc,end,result) in enumerate(owned):
                code=proc.poll()
                if code is None:
                    all_done=False
                    if time.time()>end:raise TimeoutError(f'Worker {proc.pid} reached deadline')
                elif report['workers'][i]['status']=='running':
                    report['workers'][i]['returncode']=code
                    if code:raise RuntimeError(f'Worker {proc.pid} exited {code}; retain logs')
                    value=json.loads(result.read_text())
                    assert value['status']=='complete', f'Worker {proc.pid} output incomplete'
                    report['workers'][i].update(status='complete',result=record(result))
                    save(output,report)
            if all_done:break
            if time.time()>deadline:raise TimeoutError('Launcher reached deadline')
            time.sleep(1)
        report.update(status='complete',gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc())
        raise
    finally:
        for proc,end,result in owned:stop(proc)
        for stream in streams:stream.close()
        for item,(proc,end,result) in zip(report['workers'],owned,strict=True):
            item['returncode']=proc.returncode
            if item['status']=='running':item['status']='stopped_with_launcher'
        report['seconds']=time.time()-started
        save(output,report);print(report['status'],report['seconds'],flush=True)


if __name__=='__main__':main()
