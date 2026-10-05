#!/usr/bin/env python3
"""Start the reviewed E022 trainer once the current teacher collector succeeds."""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'results/research/E022'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 out=RD/'training_queue.json';assert not out.exists()
 collect_path=RD/'collect_launcher.json';collect=json.loads(collect_path.read_text());assert collect['status'] in ('running','complete')
 trainer=ROOT/'scripts/research/train_wan_mainweight_qad_expanded.py'
 assert sha(trainer)=='920eff43be48f1daa23dc2294b546a8bec8fc9e1b167d01234d2aa307d1a00f0'
 launch=ROOT/'scripts/research/launch_wan_expanded_qad.py'
 r=dict(status='waiting_for_collection',start_epoch=time.time(),collector_pid=collect['pid'],collector_deadline=collect['deadline_epoch'],trainer_sha256=sha(trainer),launcher_sha256=sha(launch))
 def save():out.write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  while True:
   collect=json.loads(collect_path.read_text())
   if collect['status']=='complete':break
   if collect['status']!='running':raise RuntimeError('Collection failed; training was not started')
   if time.time()>r['collector_deadline']+30:raise TimeoutError('Collector launcher did not reach a terminal state')
   time.sleep(5)
  assert sha(trainer)==r['trainer_sha256'] and sha(launch)==r['launcher_sha256']
  assert not (RD/'train_launcher.json').exists() and not (RD/'train_run.json').exists()
  r.update(status='training',collection_launcher_sha256=sha(collect_path));save()
  command=[sys.executable,'-u',str(launch),'--task','train'];r['command']=command
  with (ROOT/'results/logs/E022_train_launcher.log').open('x') as log:
   proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);r['training_launcher_pid']=proc.pid;save();code=proc.wait()
  r['returncode']=code
  if code:raise RuntimeError(f'Training launcher exit {code}')
  value=json.loads((RD/'train_launcher.json').read_text());assert value['status']=='complete'
  r['status']='complete'
 except BaseException:r.update(status='failed_stop',error=traceback.format_exc());raise
 finally:r['seconds']=time.time()-r['start_epoch'];save();print(r['status'],flush=True)
if __name__=='__main__':main()
