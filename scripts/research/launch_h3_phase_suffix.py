#!/usr/bin/env python3
"""Bound E075 generation and decoding to one shared 1200-second deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E075'
check=json.loads((RD/'check.json').read_text())
assert check['status']=='complete'
idle=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
memory,usage=[int(x.strip()) for x in idle.split(',')]
assert memory<200 and usage==0, idle
assert not (RD/'launcher.json').exists()
m=json.loads((ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
env=os.environ.copy();env.update(m['environment']);env['CUDA_VISIBLE_DEVICES']='0'
env['TRITON_CACHE_DIR']='/data1/models/svdquant-wjq/research/cache/triton'
env['CUDA_CACHE_PATH']='/data1/models/svdquant-wjq/research/cache/cuda'
env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
start=time.time();deadline=start+1200
report=dict(status='running',started_unix=start,deadline_unix=deadline,phases=[])
def save():
    (RD/'launcher.json').write_text(json.dumps(report,indent=2)+'\n')
save()
for phase,python in [('run',m['python']),('decode',m['python_decode'])]:
    with (RD/f'{phase}.log').open('w') as log:
        p=subprocess.Popen([python,'-u',str(ROOT/'scripts/research/run_h3_phase_suffix.py'),
                            '--phase',phase,'--deadline-unix',str(deadline)],cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(phase=phase,pid=p.pid,started_unix=time.time());report['phases'].append(row);save()
        try:code=p.wait(timeout=max(.1,deadline-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);code=124
        row.update(exit_code=code,finished_unix=time.time());save()
        if code:
            report.update(status='failed_stop',finished_unix=time.time());save();raise SystemExit(code)
report.update(status='complete',finished_unix=time.time());save()
