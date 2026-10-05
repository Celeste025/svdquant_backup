#!/usr/bin/env python3
"""Supervise fixed E021 evaluations without changing generation or model sources."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E021'
PYTHON='/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['temporal','mjvideo'],required=True);p.add_argument('--tokenizer');a=p.parse_args()
 out=RD/(a.task+'_launcher.json');result=RD/('temporal_scores.json' if a.task=='temporal' else 'mjvideo_scores.json')
 assert not out.exists() and not result.exists()
 gen=RD/'generation_run.json';g=json.loads(gen.read_text());assert g['status']=='complete' and g['actual_totals']['videos']==16
 gpu=0 if a.task=='temporal' else 5;start=time.time();deadline=start+1800
 script=ROOT/('scripts/research/evaluate_wan_qad_temporal.py' if a.task=='temporal' else 'scripts/eval_mjvideo_rcm_vbench51.py')
 command=[PYTHON,'-u',str(script)]
 if a.task=='temporal':command+=['--deadline-unix',str(deadline)]
 else:
  assert a.tokenizer and Path(a.tokenizer).is_dir()
  command+=['--manifest',str(RD/'generation_manifest.draft.json'),'--samples','/data1/models/svdquant-wjq/research/20261003/E021','--model','/data1/models/svdquant-wjq/models/MJ-VIDEO-2B','--tokenizer',a.tokenizer,'--mjvideo-repo','/data1/models/svdquant-wjq/third_party/MJ-Video','--output',str(result),'--variants','bf16','packed_step0000','packed_step0064','svd_lr','--num-segments','8','--video-template','{variant}/{case_id}/video.mp4']
 r=dict(status='running',task=a.task,gpu=gpu,start_epoch=start,deadline_epoch=deadline,command=command,script_sha256=sha(script),protocol_sha256=sha(ROOT/'research_state/06_experiments/E021_evaluation_protocol.md'),generation_sha256=sha(gen))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_HOME='/usr/local/cuda',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MASTER_ADDR='127.0.0.1',MASTER_PORT='29563',OMP_NUM_THREADS='6')
 cache='/data1/models/svdquant-wjq/research/cache';env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
 env['PATH']=str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
 try:
  r['gpu_before']=idle_stable(gpu)
  with (ROOT/'results/logs'/('E021_'+a.task+'.log')).open('x') as log:
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
  val=json.loads(result.read_text())
  if a.task=='temporal':assert val['status']=='complete' and len(val['rows'])==16
  else:assert len(val['results'])==4 and all(set(v['variants'])=={'bf16','packed_step0000','packed_step0064','svd_lr'} for v in val['results'].values())
  r.update(status='complete',result=str(result),result_sha256=sha(result))
 except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
 finally:r['seconds']=time.time()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['seconds'],flush=True)
if __name__=='__main__':main()
