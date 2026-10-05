#!/usr/bin/env python3
"""E020 one controlled smoke then ordinary 64-update QAD run."""
import hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'results/research/E020';OUTPUT=RD/'launcher.json'
PYTHON='/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(x):OUTPUT.write_text(json.dumps(x,indent=2)+'\n')
def main():
    assert not OUTPUT.exists();started=time.time();deadline=json.loads((RD/'launcher_attempt0_failed.json').read_text())['deadline_epoch']
    files=[Path(__file__),ROOT/'scripts/research/train_wan_mainweight_qad.py',ROOT/'scripts/research/wan_mainweight_qad.py',ROOT/'scripts/research/check_wan_mainweight_qad.py',ROOT/'scripts/research/wan_native_nvfp4.py',ROOT/'scripts/research/wan_nvfp4_fastpack.py',RD/'E020_cache_inventory.json',ROOT/'research_state/06_experiments/E020_wan_mainweight_qad_plan.md']
    r=dict(status='running',start_epoch=started,deadline_epoch=deadline,wall_budget_seconds=2700,gpu=5,sources={str(p):sha(p) for p in files},stages=[])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',CUDA_HOME='/usr/local/cuda',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    cache='/data1/models/svdquant-wjq/research/cache'
    env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
    env['PATH']=str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
    try:
        r['gpu_before']=idle_stable(5)
        for name,script,args,result in [('contract','check_wan_mainweight_qad.py',[],'contract_check.json'),('train','train_wan_mainweight_qad.py',['--deadline-unix',str(deadline)],'train_run.json')]:
            assert all(sha(p)==v for p,v in r['sources'].items()),'Source changed during E020'
            assert not (RD/result).exists()
            command=[PYTHON,'-u',str(ROOT/'scripts/research'/script),*args]
            stage=dict(name=name,status='running',command=command,start_epoch=time.time());r['stages'].append(stage);save(r)
            with (ROOT/f'results/logs/E020_{name}.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);stage['pid']=proc.pid;save(r)
                try:code=proc.wait(timeout=min(300 if name=='contract' else 2700,max(.01,deadline-time.time())))
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise
            stage.update(returncode=code,seconds=time.time()-stage['start_epoch'])
            if code:raise RuntimeError(f'{name} failed with {code}')
            value=json.loads((RD/result).read_text());assert value['status']=='complete'
            stage.update(status='complete',report=str(RD/result),report_sha256=sha(RD/result));save(r)
        r['status']='complete'
    except BaseException:
        if r['stages'] and r['stages'][-1]['status']=='running':r['stages'][-1]['status']='failed_stop'
        r.update(status='failed_stop',error=traceback.format_exc());raise
    finally:r['seconds_total']=time.time()-started;save(r);print(r['status'],r['seconds_total'],flush=True)
if __name__=='__main__':main()
