#!/usr/bin/env python3
"""E022 independent data/training supervision with explicit resource budgets."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'results/research/E022'
PYTHON='/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['collect','train'],required=True);a=p.parse_args();collect=a.task=='collect'
 out=RD/(a.task+'_launcher.json');result=RD/('collect_run.json' if collect else 'train_run.json');assert not out.exists() and not result.exists()
 if not collect:
  c=json.loads((RD/'collect_launcher.json').read_text());assert c['status']=='complete' and not Path('/proc/'+str(c['pid'])).exists()
  inv=json.loads((RD/'data_inventory.json').read_text());assert inv['status']=='complete' and len(inv['train'])==256 and len(inv['validation'])==32
 gpu=0 if collect else 5;start=time.time();deadline=start+(1200 if collect else 5400)
 script=ROOT/'scripts/research'/('collect_wan_qad_training_data.py' if collect else 'train_wan_mainweight_qad_expanded.py')
 command=[PYTHON,'-u',str(script),'--deadline-unix',str(deadline)]
 if collect:command+=['--phase','collect']
 files=[Path(__file__),script,ROOT/'research_state/06_experiments/E022_expanded_qad_plan.md',RD/'data_manifest.json']
 if not collect:files+=[RD/'data_inventory.json',RD/'collect_run.json']
 r=dict(status='running',task=a.task,gpu=gpu,start_epoch=start,deadline_epoch=deadline,command=command,sources={str(f):sha(f) for f in files})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_HOME='/usr/local/cuda',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',SVDQUANT_DATA_ROOT='/data1/models/svdquant-wjq',OMP_NUM_THREADS='6')
 cache='/data1/models/svdquant-wjq/research/cache';env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
 env['PATH']=str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
 try:
  r['gpu_before']=idle_stable(gpu)
  with (ROOT/'results/logs'/('E022_'+a.task+'.log')).open('x') as log:
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['pid']=proc.pid;out.write_text(json.dumps(r,indent=2)+'\n')
   try:code=proc.wait(timeout=max(.01,deadline-time.time()))
   except BaseException:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  r['returncode']=code
  if code:raise RuntimeError(f'{a.task} exited {code}')
  value=json.loads(result.read_text());assert value['status']=='complete';r.update(status='complete',result=str(result),result_sha256=sha(result))
 except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
 finally:r['seconds']=time.time()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['seconds'],flush=True)
if __name__=='__main__':main()
