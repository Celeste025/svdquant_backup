#!/usr/bin/env python3
"""Execute frozen E012 phases serially; stop before native on teacher-gate failure."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/wjq/workspace/svdquant-exp')
PYTHON = '/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python'
RUNNER = ROOT/'scripts/research/probe_h3_conditional_response.py'
CHECK = ROOT/'results/research/E012_h3_check.json'
REPORT = ROOT/'results/research/E012_launcher.json'
MANIFEST = ROOT/'research_state/06_experiments/E012_h3_conditional_response_manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def idle_gpu():
    output = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, used, utilization = [v.strip() for v in output.strip().split(',')]
    if int(used) > 64 or int(utilization) > 1:
        raise RuntimeError(f'GPU0 not idle: {output.strip()}')
    return {'gpu': 0, 'uuid': uuid, 'used_mib': int(used), 'utilization_percent': int(utilization)}


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    check = json.loads(CHECK.read_text())
    if check['status'] != 'complete' or check['cuda_initialized'] is not False:
        raise RuntimeError('A successful CPU-only check is required')
    for path, info in check['sources'].items():
        if sha(path) != info['sha256']:
            raise RuntimeError(f'Source changed after CPU check: {path}')
    budget = json.loads(MANIFEST.read_text())['budgets']['total_wall_seconds']
    record = {'experiment': 'E012', 'status': 'running', 'gpu': 0,
              'launcher_sha256': sha(__file__), 'runner_sha256': sha(RUNNER),
              'cpu_check_sha256': sha(CHECK), 'wall_budget_seconds': budget, 'stages': []}
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(PYTHON).parent)+':'+env.get('PATH', '')
    try:
        for phase in ('prepare', 'bf16', 'native'):
            remaining = budget-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('E012 total wall budget exhausted')
            if phase == 'native':
                teacher = json.loads((ROOT/'results/research/E012_h3_bf16.json').read_text())
                if teacher['status'] == 'stopped_teacher_gate':
                    if teacher['teacher_gate']['pass']:
                        raise RuntimeError('Inconsistent teacher gate result')
                    record['status'] = 'stopped_teacher_gate'
                    record['native_skipped'] = True
                    break
                if teacher['status'] != 'complete' or not teacher['teacher_gate']['pass']:
                    raise RuntimeError('Teacher gate not passed')
            gpu = idle_gpu()
            command = [PYTHON, '-u', str(RUNNER), '--phase', phase]
            log = ROOT/'results/logs'/f'E012_h3_{phase}.log'
            entry = {'phase': phase, 'command': command, 'gpu_before': gpu,
                     'log': str(log), 'status': 'running'}
            record['stages'].append(entry)
            REPORT.write_text(json.dumps(record, indent=2)+'\n')
            print(f'Starting {phase}; log={log}', flush=True)
            before = time.monotonic()
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=remaining)
            entry.update(returncode=result.returncode, seconds=time.monotonic()-before,
                         status='complete' if result.returncode == 0 else 'failed_stop')
            if result.returncode != 0:
                raise RuntimeError(f'{phase} failed: {result.returncode}')
            phase_path = ROOT/'results/research'/f'E012_h3_{phase}.json'
            phase_report = json.loads(phase_path.read_text())
            if phase_report['status'] not in ('complete', 'stopped_teacher_gate'):
                raise RuntimeError(f'{phase}: unexpected status {phase_report["status"]}')
            entry['report'] = str(phase_path)
            entry['report_sha256'] = sha(phase_path)
            print(f'Completed {phase}: {phase_report["status"]}', flush=True)
        else:
            record['status'] = 'complete'
    except BaseException as exc:
        if record['stages'] and record['stages'][-1]['status'] == 'running':
            record['stages'][-1]['status'] = 'failed_stop'
        record.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['seconds_total'] = time.monotonic()-started
        REPORT.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
