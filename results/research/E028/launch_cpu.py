import json, os, pathlib, subprocess, time
root=pathlib.Path('/home/wjq/workspace/svdquant-exp')
out=root/'results/research/E028'
start=time.time()
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
cmd=['/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python','scripts/research/probe_h3_center_spectrum.py','--phase','run','--deadline-unix',str(start+299)]
record={'start_unix':start,'deadline_unix':start+300,'command':cmd,'supervisor_pid':os.getpid(),'status':'running'}
with (out/'run.log').open('w') as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 record['worker_pid']=p.pid;(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
 try: record['returncode']=p.wait(timeout=300)
 except subprocess.TimeoutExpired: p.kill();p.wait();record['status']='timeout'
 else: record['status']='complete' if record['returncode']==0 else 'failed'
record['elapsed_s']=time.time()-start
(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
