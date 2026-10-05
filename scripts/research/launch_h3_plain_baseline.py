#!/usr/bin/env python3
"""E014 serial stages with one immutable 45-minute wall deadline."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

ROOT = Path('/home/wjq/workspace/svdquant-exp')
PYTHON = '/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python'
RD = ROOT/'results/research/E014'
RUNNER = ROOT/'scripts/research/probe_h3_plain_baseline.py'
EXPORTER = ROOT/'scripts/research/export_h3_plain_nvfp4.py'
CHECK = RD/'E014_check_bf16.json'
FREEZE = RD/'frozen_contract.json'
OUTPUT = RD/'launcher.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def complete(path):
    x = json.loads(Path(path).read_text())
    if x['status'] != 'complete':
        raise RuntimeError(f'Incomplete prerequisite: {path}')
    return x

def idle(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, memory, use = [v.strip() for v in row.strip().split(',')]
    if int(memory) > 64 or int(use) > 1:
        raise RuntimeError(f'GPU{gpu} not idle: {row.strip()}')
    return dict(index=gpu, uuid=uuid, memory_mib=int(memory), utilization_percent=int(use))

def main():
    if OUTPUT.exists():
        raise FileExistsError('Do not overwrite or reset a previous E014 run')
    freeze = complete(FREEZE)
    for path, digest in freeze['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen input/source changed: {path}')
    complete(CHECK)
    complete(RD/'export_cpu_check.json')
    started = time.time()
    deadline = started+2700
    record = dict(experiment='E014', status='running', start_epoch=started,
        deadline_epoch=deadline, wall_budget_seconds=2700, max_complete_dit_calls=30,
        planned_dit_calls=24, freeze_sha256=sha(FREEZE), stages=[])
    stages = [('export', 0, EXPORTER, ['--phase', 'export'], RD/'export.json', 0)]
    for phase, calls in [('evaluate', 3), ('bench', 5)]:
        for arm in ['bf16', 'svd', 'plain']:
            stages.append((phase+'_'+arm, 5, RUNNER,
                ['--phase', phase, '--arm', arm, '--check-report', str(CHECK)],
                RD/f'E014_{phase}_{arm}.json', calls))
    try:
        for name, gpu, script, args, result_path, calls in stages:
            available = deadline-time.time()
            if available <= 0:
                raise TimeoutError('Original E014 wall budget exhausted')
            gpu_before = idle(gpu)
            if result_path.exists():
                raise FileExistsError(result_path)
            command = [PYTHON, '-u', str(script), *args, '--deadline-unix', str(deadline)]
            log = ROOT/'results/logs'/f'E014_{name}.log'
            entry = dict(name=name, status='running', command=command, log=str(log),
                         gpu_before=gpu_before, planned_dit_calls=calls)
            record['stages'].append(entry)
            OUTPUT.write_text(json.dumps(record, indent=2)+'\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
            env['PATH'] = str(Path(PYTHON).parent)+':'+env.get('PATH', '')
            before = time.time()
            print(f'Start {name} GPU{gpu}; {log}', flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
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
            entry.update(returncode=code, seconds=time.time()-before)
            if code:
                raise RuntimeError(f'{name} failed: returncode={code}')
            complete(result_path)
            entry.update(status='complete', report=str(result_path), report_sha256=sha(result_path))
            print(f'Complete {name} in {entry["seconds"]:.1f}s', flush=True)
        record['status'] = 'complete'
    except BaseException as exc:
        if record['stages'] and record['stages'][-1]['status'] == 'running':
            record['stages'][-1]['status'] = 'failed_stop'
        record.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['seconds_total'] = time.time()-started
        OUTPUT.write_text(json.dumps(record, indent=2)+'\n')

if __name__ == '__main__':
    main()
