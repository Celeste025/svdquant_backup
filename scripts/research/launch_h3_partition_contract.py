#!/usr/bin/env python3
"""One E019 native contract audit, never a generation or multi-GPU job."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from launch_h3_plain_baseline import ROOT, sha, complete
from resume_h3_plain_baseline import idle_stable

RD = ROOT/'results/research/E019'
OUTPUT = RD/'launcher.json'

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = complete(RD/'frozen_contract.json')
    for path,digest in frozen['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen file changed: {path}')
    if complete(RD/'probe_check.json').get('cuda_initialized') is not False:
        raise RuntimeError('Missing CPU-only packet validation')
    prior = json.loads((ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = prior['python']
    start = time.time()
    deadline = start+900
    report = dict(experiment='E019',status='running',start_epoch=start,deadline_epoch=deadline,
        wall_budget_seconds=900,gpu=5,max_dit_calls=0,max_attention_calls=12,
        freeze_sha256=sha(RD/'frozen_contract.json'),stages=[])
    try:
        before = idle_stable(5)
        result_path = RD/'probe_run.json'
        if result_path.exists():
            raise FileExistsError(result_path)
        cmd = [python,'-u',str(ROOT/'scripts/research/probe_h3_partition_contract.py'),
               '--phase','run','--deadline-unix',str(deadline)]
        log = ROOT/'results/logs/E019_probe.log'
        env = dict(os.environ,**prior['environment'],CUDA_VISIBLE_DEVICES='5',PYTHONUNBUFFERED='1')
        env['PATH'] = str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
        env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
        env['CUDA_CACHE_PATH'] = '/data1/models/svdquant-wjq/research/cache/cuda'
        stage = dict(name='probe',status='running',command=cmd,log=str(log),gpu_before=before,start_epoch=time.time())
        report['stages'].append(stage)
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
        print('E019 start 12-call partition audit',flush=True)
        with log.open('x') as stream:
            proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            stage['pid'] = proc.pid
            OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
            try:
                code = proc.wait(timeout=max(.01,deadline-time.time()))
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
                raise
        stage.update(returncode=code,seconds=time.time()-stage['start_epoch'])
        if code:
            raise RuntimeError(f'E019 probe exited {code}')
        result = complete(result_path)
        if result['attention_calls'] != 12 or result['complete_dit_calls'] != 0 or len(result['cases']) != 3:
            raise RuntimeError('Actual call allocation differs')
        stage.update(status='complete',report=str(result_path),report_sha256=sha(result_path))
        report['status'] = 'complete'
        print(f'E019 complete in {stage["seconds"]:.1f}s',flush=True)
    except BaseException as exc:
        if report['stages'] and report['stages'][-1]['status']=='running':
            report['stages'][-1]['status']='failed_stop'
        report.update(status='failed_stop',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.time()-start
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
