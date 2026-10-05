import json,os,subprocess,time
from pathlib import Path
rd=Path(__file__).resolve().parent
r=json.loads((rd/'launch.json').read_text())
env=os.environ.copy();env.update(r['environment'])
with Path(r['log']).open('xb') as log:
 p=subprocess.Popen(r['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
 (rd/'supervisor_started.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),timeout_pid=p.pid,start_epoch=time.time()),indent=2))
 status=p.wait()
 (rd/'process_exit.json').write_text(json.dumps(dict(exit_code=status,end_epoch=time.time(),deadline_unix=r['deadline_unix'],timeout_exit=(status==124),command=r['command']),indent=2))
