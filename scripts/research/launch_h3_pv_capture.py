#!/usr/bin/env python3
import argparse,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/research/E080/v2';RUNNER=ROOT/'scripts/research/capture_h3_pv_contrast_v2.py'
p=argparse.ArgumentParser();p.add_argument('--replica',type=int,choices=[0,1],required=True);r=p.parse_args().replica
check=json.loads((OUT/f'check_r{r}.json').read_text());assert check['status']=='complete' and check['cuda_initialized'] is False
memory,util=subprocess.check_output(['nvidia-smi','-i',str(r),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');assert int(memory)<200 and int(util)==0
start=time.time();deadline=start+900;result=dict(status='running',replica=r,started_unix=start,deadline_unix=deadline);dest=OUT/f'launcher_r{r}.json';assert not dest.exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(r),PYTHONUNBUFFERED='1',CUDA_HOME='/usr/local/cuda',FLASHINFER_WORKSPACE_BASE='/data1/models/svdquant-wjq/research/cache/flashinfer',TMPDIR='/data1/models/svdquant-wjq/research/cache/tmp',TORCH_EXTENSIONS_DIR='/data1/models/svdquant-wjq/research/cache/torch_extensions',MAX_JOBS='4',TRITON_CACHE_DIR='/data1/models/svdquant-wjq/research/cache/triton',CUDA_CACHE_PATH='/data1/models/svdquant-wjq/research/cache/cuda')
env['PATH']='/usr/local/cuda/bin:/home/wjq/.conda/envs/convrot-wan/bin:'+env['PATH']
try:
 with (OUT/f'capture_r{r}.log').open('x') as stream:
  child=subprocess.Popen(['/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python','-u',str(RUNNER),'--phase','capture','--replica',str(r),'--deadline-unix',str(deadline)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  result['pid']=child.pid;dest.write_text(json.dumps(result,indent=2))
  try:code=child.wait(timeout=900)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('Capture wall budget exhausted')
  result['exit_code']=code;assert code==0
  record=json.loads((OUT/f'capture_r{r}.json').read_text());assert record['status']=='complete' and record['final_replay_exact']
 result['status']='complete'
except BaseException as e:result.update(status='failed_stop',error=repr(e));raise
finally:result['finished_unix']=time.time();dest.write_text(json.dumps(result,indent=2))
