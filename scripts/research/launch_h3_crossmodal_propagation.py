#!/usr/bin/env python3
"""E015: one shared deadline for three serial, independently loaded arms."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from launch_h3_plain_baseline import ROOT, PYTHON, sha, complete
from resume_h3_plain_baseline import idle_stable

RD = ROOT/'results/research/E015'
FREEZE = RD/'frozen_contract.json'
OUTPUT = RD/'launcher.json'
RUNNER = ROOT/'scripts/research/probe_h3_crossmodal_propagation.py'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = complete(FREEZE)
    for path, digest in frozen['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen input/source changed: {path}')
    complete(RD/'E015_prepare.json')
    complete(RD/'check.json')
    start = time.time()
    deadline = start+1200
    report = dict(experiment='E015', status='running', start_epoch=start,
        deadline_epoch=deadline, wall_budget_seconds=1200, max_complete_dit_calls=20,
        planned_dit_calls=18, freeze_sha256=sha(FREEZE), stages=[])
    try:
        for arm, calls in [('bf16', 2), ('svd', 8), ('plain', 8)]:
            gpu = idle_stable(5)
            available = deadline-time.time()
            if available <= 0:
                raise TimeoutError('Original E015 wall deadline expired')
            result_path = RD/f'evaluate_{arm}.json'
            if result_path.exists():
                raise FileExistsError(result_path)
            command = [PYTHON, '-u', str(RUNNER), '--phase', 'evaluate', '--arm', arm,
                       '--deadline-unix', str(deadline)]
            log = ROOT/'results/logs'/f'E015_{arm}.log'
            entry = dict(name=arm, status='running', command=command, log=str(log),
                gpu_before=gpu, planned_dit_calls=calls, start_epoch=time.time())
            report['stages'].append(entry)
            OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', PYTHONUNBUFFERED='1')
            env['PATH'] = str(Path(PYTHON).parent)+':'+env.get('PATH', '')
            print(f'Start {arm} GPU5; {log}', flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = proc.pid
                OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
                try:
                    code = proc.wait(timeout=available)
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    raise
            entry.update(returncode=code, end_epoch=time.time())
            entry['seconds'] = entry['end_epoch']-entry['start_epoch']
            if code:
                raise RuntimeError(f'{arm} failed: returncode={code}')
            result = complete(result_path)
            if result['complete_dit_calls'] != calls:
                raise RuntimeError(f'{arm}: unexpected DiT call count')
            entry.update(status='complete', report=str(result_path), report_sha256=sha(result_path))
            print(f'Complete {arm} in {entry["seconds"]:.1f}s', flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        if report['stages'] and report['stages'][-1]['status'] == 'running':
            report['stages'][-1]['status'] = 'failed_stop'
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.time()-start
        OUTPUT.write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
