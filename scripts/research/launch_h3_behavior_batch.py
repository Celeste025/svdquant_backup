#!/usr/bin/env python3
"""E013 bounded batches; one unchanged deadline includes manual review intervals."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import threading
import traceback

ROOT = Path('/home/wjq/workspace/svdquant-exp')
PYTHON = '/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python'
RUNNER = ROOT/'scripts/research/run_h3_behavior_probe.py'
REPORT_DIR = ROOT/'results/research/E013'
FREEZE = REPORT_DIR/'frozen_contract.json'
BUDGET = REPORT_DIR/'wall_budget.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')

def complete(path):
    value = json.loads(path.read_text())
    if value['status'] != 'complete':
        raise RuntimeError(f'Incomplete prerequisite: {path}')
    return value

def idle(gpu):
    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, memory, use = [v.strip() for v in row.strip().split(',')]
    if int(memory) > 64 or int(use) > 1:
        raise RuntimeError(f'GPU{gpu} not idle: {row.strip()}')
    return dict(gpu=gpu, uuid=uuid, memory_mib=int(memory), utilization=int(use))

def invoke(phase, arm, case, gpu, deadline, stop):
    if stop.is_set():
        raise RuntimeError("Another case failed; do not start another stage")
    before_gpu = idle(gpu)
    left = deadline-time.time()
    if left <= 0:
        raise TimeoutError('Original E013 90-minute deadline exhausted')
    command = [PYTHON, '-u', str(RUNNER), '--phase', phase]
    if case is not None:
        command += ['--arm', arm, '--case-ids', str(case)]
    name = phase if case is None else f'{phase}_{arm}_p{case:03d}'
    log = ROOT/'results/logs'/f'E013_{name}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(PYTHON).parent)+':'+env.get('PATH', '')
    started = time.time()
    print(f'Start {name} GPU{gpu}: {log}', flush=True)
    with log.open('x') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                if stop.is_set():
                    raise RuntimeError('Another E013 case failed; stop this process group')
                left = deadline-time.time()
                if left <= 0:
                    raise TimeoutError('Original E013 deadline exhausted')
                try:
                    code = proc.wait(timeout=min(1., left))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise RuntimeError(f'{name} failed: {code}; preserved log={log}')
    report = REPORT_DIR/f'E013_h3_{phase}.json' if case is None else (
        REPORT_DIR/f'p{case:03d}'/f'E013_h3_{phase}_{arm}.json')
    value = complete(report)
    print(f'Complete {name}: {time.time()-started:.1f}s', flush=True)
    return dict(phase=phase, arm=arm, case=case, gpu_before=before_gpu, command=command,
                seconds=time.time()-started, report=str(report), report_sha256=sha(report),
                peak_allocated_gib=value.get('peak_allocated_gib_process'), log=str(log))

def case_stages(case, gpu, arm, deadline, stop):
    try:
        return [invoke('denoise', arm, case, gpu, deadline, stop),
                invoke('decode', arm, case, gpu, deadline, stop)]
    except BaseException:
        stop.set()
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, choices=['prepare', 'teacher1', 'teacher2', 'native1', 'native2'])
    args = parser.parse_args()
    output = REPORT_DIR/f'launcher_{args.batch}.json'
    if output.exists():
        raise FileExistsError(output)
    freeze = complete(FREEZE)
    for path, digest in freeze['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen source changed: {path}')
    check = complete(REPORT_DIR/'E013_h3_check.json')
    if check['cuda_initialized'] is not False:
        raise RuntimeError('Preflight was not CPU-only')
    for path, info in check['sources'].items():
        if sha(path) != info['sha256']:
            raise RuntimeError(f'CPU-checked source changed: {path}')
    if args.batch == 'prepare':
        if BUDGET.exists():
            raise FileExistsError('Never reset the global wall budget')
        idle(0)
        write(BUDGET, dict(start_epoch=time.time(), wall_budget_seconds=5400))
    budget = json.loads(BUDGET.read_text())
    deadline = budget['start_epoch']+budget['wall_budget_seconds']
    if args.batch not in ('prepare', 'teacher1'):
        gate_path = REPORT_DIR/('teacher_seed1_gate.json' if args.batch == 'teacher2' else 'teacher_all_gate.json')
        gate = complete(gate_path)
        if gate['pass'] is not True:
            raise RuntimeError('Teacher behavior gate did not pass')
        for path, digest in gate['evidence'].items():
            if sha(path) != digest:
                raise RuntimeError(f'Gate evidence changed: {path}')
    if args.batch == 'native2':
        complete(REPORT_DIR/'launcher_native1.json')
    record = dict(experiment='E013', batch=args.batch, status='running',
                  freeze_sha256=sha(FREEZE), budget=budget, stages=[])
    write(output, record)
    started = time.time()
    stop = threading.Event()
    try:
        if args.batch == 'prepare':
            record['stages'] = [invoke('prepare', None, None, 0, deadline, stop)]
        else:
            complete(REPORT_DIR/'E013_h3_prepare.json')
            ids = [1, 2] if args.batch in ('teacher1', 'native1') else [3, 4]
            arm = 'native' if args.batch.startswith('native') else 'bf16'
            idle(0)
            idle(5)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(case_stages, case, gpu, arm, deadline, stop)
                           for case, gpu in zip(ids, [0, 5])]
                errors = []
                for future in as_completed(futures):
                    try:
                        record['stages'].extend(future.result())
                    except BaseException as exc:
                        errors.append(repr(exc))
                if errors:
                    raise RuntimeError('; '.join(errors))
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds_total=time.time()-started,
                      elapsed_since_first_gpu=time.time()-budget['start_epoch'])
        write(output, record)

if __name__ == '__main__':
    main()
