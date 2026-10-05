#!/usr/bin/env python3
"""Independent E020 timing or fixed E021 generation, only after training exits."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()
from resume_h3_plain_baseline import idle_stable
PYTHON='/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python'
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['bench','generation'],required=True);args=p.parse_args()
    gen=args.task=='generation';rd=ROOT/('results/research/E021' if gen else 'results/research/E020');output=rd/('generation_launcher.json' if gen else 'deployment_launcher.json')
    assert not output.exists();train=json.loads((ROOT/'results/research/E020/train_run.json').read_text());assert train['status']=='complete'
    launch=json.loads((ROOT/'results/research/E020/launcher.json').read_text());assert launch['status']=='complete' and not Path('/proc/'+str(launch['stages'][-1]['pid'])).exists()
    script=ROOT/'scripts/research'/('generate_wan_qad_comparison.py' if gen else 'bench_wan_qad_deployment.py')
    result=rd/('generation_run.json' if gen else 'deployment_bench.json');assert not result.exists()
    gpu=0 if gen else 5;started=time.time();deadline=started+(1800 if gen else 1200)
    r=dict(status='running',task=args.task,gpu=gpu,start_epoch=started,deadline_epoch=deadline,script=str(script),script_sha256=sha(script),training_report_sha256=sha(ROOT/'results/research/E020/train_run.json'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_HOME='/usr/local/cuda',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',SVDQUANT_DATA_ROOT='/data1/models/svdquant-wjq',RCM_RUNS_ROOT='/data1/models/svdquant-wjq/research/20261003/'+('E021_scratch' if gen else 'E020_bench_scratch'))
    cache='/data1/models/svdquant-wjq/research/cache';env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
    env['PATH']=str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
    try:
        r['gpu_before']=idle_stable(gpu)
        if gen:
            for row in train['exports']:assert sha(row['file'])==row['sha256']
        command=[PYTHON,'-u',str(script),'--deadline-unix',str(deadline)];r['command']=command
        log=ROOT/'results/logs'/('E021_generation.log' if gen else 'E020_deployment.log')
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);r['pid']=proc.pid;output.write_text(json.dumps(r,indent=2)+'\n')
            try:code=proc.wait(timeout=max(.01,deadline-time.time()))
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        r['returncode']=code
        if code:raise RuntimeError(f'{args.task} exited {code}')
        value=json.loads(result.read_text());assert value['status']=='complete';r.update(status='complete',result=str(result),result_sha256=sha(result))
    except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
    finally:r['seconds_total']=time.time()-started;output.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['seconds_total'],flush=True)
if __name__=='__main__':main()
