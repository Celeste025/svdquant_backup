#!/usr/bin/env python3
"""E022 fixed baseline or development-selected videos, supervised on GPU0."""
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
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['baselines','selected'],required=True);a=p.parse_args()
 out=RD/('video_'+a.phase+'_launcher.json');result=RD/('video_'+a.phase+'.json');assert not out.exists() and not result.exists()
 script=ROOT/'scripts/research/generate_wan_qad_expanded_comparison.py';manifest=RD/'video_test_manifest.json'
 if a.phase=='selected':
  t=json.loads((RD/'train_launcher.json').read_text());assert t['status']=='complete' and not Path('/proc/'+str(t['pid'])).exists()
  b=json.loads((RD/'video_baselines_launcher.json').read_text());assert b['status']=='complete'
 else:
  c=json.loads((RD/'video_baselines_cpucheck.json').read_text());assert c['status']=='complete' and c['source']['sha256']==sha(script) and c['manifest_reference']['sha256']==sha(manifest)
 start=time.time();deadline=start+(1200 if a.phase=='baselines' else 600)
 command=[PYTHON,'-u',str(script),'--phase',a.phase,'--deadline-unix',str(deadline)]
 r=dict(status='running',phase=a.phase,gpu=0,start_epoch=start,deadline_epoch=deadline,command=command,sources={str(f):sha(f) for f in [Path(__file__),script,manifest,ROOT/'research_state/06_experiments/E022_video_execution_plan.md']})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',SVDQUANT_DATA_ROOT='/data1/models/svdquant-wjq',OMP_NUM_THREADS='6')
 cache='/data1/models/svdquant-wjq/research/cache';env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
 env['PATH']=str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
 try:
  r['gpu_before']=idle_stable(0)
  with (ROOT/'results/logs'/('E022_video_'+a.phase+'.log')).open('x') as log:
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['pid']=proc.pid;out.write_text(json.dumps(r,indent=2)+'\n')
   try:code=proc.wait(timeout=max(.01,deadline-time.time()))
   except BaseException:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  r['returncode']=code
  if code:raise RuntimeError(f'{a.phase} exited {code}')
  value=json.loads(result.read_text());assert value['status']=='complete';r.update(status='complete',result=str(result),result_sha256=sha(result),actual_totals=value['actual_totals'])
 except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
 finally:r['seconds']=time.time()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['seconds'],flush=True)
if __name__=='__main__':main()
