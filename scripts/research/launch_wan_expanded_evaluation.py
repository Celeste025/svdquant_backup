#!/usr/bin/env python3
"""E022 fixed per-phase temporal or MJ evaluation, local assets only."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'results/research/E022'
PY='/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['baselines','selected'],required=True);p.add_argument('--metric',choices=['temporal','mjvideo'],required=True);a=p.parse_args()
 out=RD/(a.metric+'_'+a.phase+'_launcher.json');result=RD/(a.metric+'_'+a.phase+'.json');assert not out.exists() and not result.exists()
 generation=RD/('video_'+a.phase+'.json');g=json.loads(generation.read_text());assert g['status']=='complete'
 validation=RD/('video_'+a.phase+'_validation.json');v=json.loads(validation.read_text());assert v['status']=='complete'
 temporal=a.metric=='temporal';gpu=0 if temporal else 1;start=time.time();deadline=start+(1800 if temporal else 600)
 script=ROOT/'scripts/research'/('evaluate_wan_qad_expanded_temporal.py' if temporal else 'eval_mjvideo_e021.py')
 command=[PY,'-u',str(script)]
 if temporal:command+=['--phase',a.phase,'--deadline-unix',str(deadline)]
 else:
  arms=['bf16','plain_step0000','svd_lr'] if a.phase=='baselines' else ['qad_native_dev_selected']
  command+=['--manifest',str(RD/'mjvideo_input_manifest.json'),'--samples','/data1/models/svdquant-wjq/research/20261003/E022/videos','--model','/data1/models/svdquant-wjq/models/MJ-VIDEO-2B','--tokenizer','/data1/models/svdquant-wjq/research/20261003/E021/tokenizer','--mjvideo-repo','/data1/models/svdquant-wjq/third_party/MJ-Video','--output',str(result),'--variants',*arms,'--num-segments','8','--video-template','{variant}/{case_id}/video.mp4']
 r=dict(status='running',phase=a.phase,metric=a.metric,gpu=gpu,start_epoch=start,deadline_epoch=deadline,command=command,script_sha256=sha(script),generation_sha256=sha(generation),validation_sha256=sha(validation))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_HOME='/usr/local/cuda',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',MASTER_ADDR='127.0.0.1',MASTER_PORT='29564',OMP_NUM_THREADS='6')
 cache='/data1/models/svdquant-wjq/research/cache';env.update(TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda',TORCH_EXTENSIONS_DIR=cache+'/torch_extensions',TMPDIR=cache+'/tmp',MAX_JOBS='4')
 env['PATH']=str(Path(PY).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
 try:
  r['gpu_before']=idle_stable(gpu)
  with (ROOT/'results/logs'/('E022_'+a.metric+'_'+a.phase+'.log')).open('x') as log:
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);r['pid']=proc.pid;out.write_text(json.dumps(r,indent=2)+'\n')
   try:code=proc.wait(timeout=max(.01,deadline-time.time()))
   except BaseException:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  r['returncode']=code
  if code:raise RuntimeError(f'{a.metric}/{a.phase} exited {code}')
  value=json.loads(result.read_text());n=48 if a.phase=='baselines' else 16
  if temporal:assert value['status']=='complete' and len(value['rows'])==n
  else:assert len(value['results'])==16 and all(set(c['variants'])==set(arms) for c in value['results'].values())
  r.update(status='complete',result=str(result),result_sha256=sha(result))
 except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
 finally:r['seconds']=time.time()-start;out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['seconds'],flush=True)
if __name__=='__main__':main()
