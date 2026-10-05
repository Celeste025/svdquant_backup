#!/usr/bin/env python3
"""Fix only missing decoder dependency; retain failed E021 MJ wall deadline."""
import hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'results/research/E021'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 old=json.loads((RD/'mjvideo_launcher.json').read_text());assert old['status']=='failed_stop' and old['returncode']==1
 assert not Path('/proc/'+str(old['pid'])).exists()
 result=RD/'mjvideo_scores.json';out=RD/'mjvideo_launcher_retry1.json';assert not result.exists() and not out.exists()
 deadline=old['deadline_epoch'];assert time.time()<deadline
 script=ROOT/'scripts/research/eval_mjvideo_e021.py';command=list(old['command']);command[2]=str(script)
 r=dict(status='running',gpu=5,start_epoch=time.time(),deadline_epoch=deadline,command=command,source_sha256=sha(script),prior_launcher_sha256=sha(RD/'mjvideo_launcher.json'),fix='Append existing VBench site-packages for missing decord; same official evaluator math; record actual sample indices and model attention fallback.')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',CUDA_HOME='/usr/local/cuda',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MASTER_ADDR='127.0.0.1',MASTER_PORT='29563',OMP_NUM_THREADS='6')
 cache='/data1/models/svdquant-wjq/research/cache';env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
 env['PATH']=str(Path(command[0]).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
 try:
  r['gpu_before']=idle_stable(5)
  with (ROOT/'results/logs/E021_mjvideo_retry1.log').open('x') as log:
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['pid']=proc.pid;out.write_text(json.dumps(r,indent=2)+'\n')
   try:code=proc.wait(timeout=max(.01,deadline-time.time()))
   except BaseException:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  r['returncode']=code
  if code:raise RuntimeError(f'MJ evaluator exit {code}')
  val=json.loads(result.read_text());assert len(val['results'])==4 and all(set(v['variants'])=={'bf16','packed_step0000','packed_step0064','svd_lr'} for v in val['results'].values())
  r.update(status='complete',result_sha256=sha(result))
 except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
 finally:r['seconds']=time.time()-r['start_epoch'];out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['seconds'],flush=True)
if __name__=='__main__':main()
