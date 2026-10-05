#!/usr/bin/env python3
"""E018: one capture and 27 fixed attention probes under one 900s deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from launch_h3_plain_baseline import ROOT, sha, complete
from resume_h3_plain_baseline import idle_stable

RD = ROOT/'results/research/E018'
OUTPUT = RD/'launcher.json'

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = complete(RD/'frozen_contract.json')
    for path,digest in frozen['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen file changed: {path}')
    for name in ('capture_check','probe_check'):
        if complete(RD/f'{name}.json').get('cuda_initialized') is not False:
            raise RuntimeError('Missing CPU-only validation')
    prior = json.loads((ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = prior['python']
    start = time.time()
    deadline = start+900
    report = dict(experiment='E018',status='running',start_epoch=start,deadline_epoch=deadline,
        wall_budget_seconds=900,gpu=5,max_dit_calls=1,max_attention_probes=27,
        freeze_sha256=sha(RD/'frozen_contract.json'),stages=[])
    try:
        for name,script,phase in [('capture','capture_h3_query_phase.py','capture'),
                                  ('probe','probe_h3_query_phase.py','run')]:
            before = idle_stable(5)
            remaining = deadline-time.time()
            if remaining <= 0:
                raise TimeoutError('Original E018 deadline expired')
            result_path = RD/f'{name}_run.json'
            if result_path.exists():
                raise FileExistsError(result_path)
            cmd = [python,'-u',str(ROOT/'scripts/research'/script),'--phase',phase,
                   '--deadline-unix',str(deadline)]
            log = ROOT/'results/logs'/f'E018_{name}.log'
            env = dict(os.environ,**prior['environment'],CUDA_VISIBLE_DEVICES='5',PYTHONUNBUFFERED='1')
            env['PATH'] = str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
            env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
            env['CUDA_CACHE_PATH'] = '/data1/models/svdquant-wjq/research/cache/cuda'
            entry = dict(name=name,status='running',command=cmd,log=str(log),gpu_before=before,start_epoch=time.time())
            report['stages'].append(entry)
            OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
            print(f'E018 start {name}',flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                entry['pid'] = proc.pid
                OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
                try:
                    code = proc.wait(timeout=remaining)
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL)
                            proc.wait()
                    raise
            entry.update(returncode=code,seconds=time.time()-entry['start_epoch'])
            if code:
                raise RuntimeError(f'{name} exited {code}')
            result = complete(result_path)
            if name == 'capture':
                if not (result['attempted_dit_calls'] == result['complete_dit_calls'] == 1
                        and result['raw_replay_exact'] and result['velocity_replay_exact']
                        and len(result['cases']) == 3):
                    raise RuntimeError('Incomplete or inexact source replay')
            elif not (result['attention_calls'] == 27 and len(result['cases']) == 3
                      and all(len(c['outputs']) == 9 for c in result['cases'])):
                raise RuntimeError('Attention allocation changed')
            entry.update(status='complete',report=str(result_path),report_sha256=sha(result_path))
            print(f'E018 complete {name}: {entry["seconds"]:.1f}s',flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        if report['stages'] and report['stages'][-1]['status'] == 'running':
            report['stages'][-1]['status'] = 'failed_stop'
        report.update(status='failed_stop',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.time()-start
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
