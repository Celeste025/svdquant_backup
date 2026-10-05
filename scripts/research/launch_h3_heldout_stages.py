#!/usr/bin/env python3
"""Run the frozen E010 stages serially on an otherwise idle GPU0."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/wjq/workspace/svdquant-exp')
PYTHON = '/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python'
RUNNER = ROOT/'scripts/research/run_h3_native_paired_video.py'
REPORT = ROOT/'results/research/E010_launcher.json'
CHECK = ROOT/'results/research/E010_h3_check.json'
STAGES = [('prepare', []), ('denoise_bf16', ['--arm', 'bf16']),
          ('denoise_native', ['--arm', 'native']), ('decode', [])]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def idle_gpu():
    output = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, used, utilization = [v.strip() for v in output.strip().split(',')]
    if int(used) > 64 or int(utilization) > 1:
        raise RuntimeError(f'GPU0 is no longer idle: {output.strip()}')
    return {'gpu': 0, 'uuid': uuid, 'used_mib': int(used), 'utilization_percent': int(utilization)}


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    check = json.loads(CHECK.read_text())
    if check['status'] != 'complete' or check['cuda_initialized'] is not False:
        raise RuntimeError('CPU-only preflight must pass first')
    for path, info in check['sources'].items():
        if sha(Path(path)) != info['sha256']:
            raise RuntimeError(f'Source changed after CPU check: {path}')
    record = {'experiment': 'E010', 'status': 'running', 'gpu': 0,
              'source_sha256': sha(Path(__file__)), 'runner_sha256': sha(RUNNER),
              'cpu_check_sha256': sha(CHECK), 'stages': [], 'wall_budget_seconds': 3600}
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(PYTHON).parent)+':'+env.get('PATH', '')
    try:
        for name, extra in STAGES:
            remaining = 3600 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('E010 total wall budget exhausted')
            gpu = idle_gpu()
            command = [PYTHON, '-u', str(RUNNER), '--phase', name.split('_')[0], *extra]
            log = ROOT/'results/logs'/f'E010_h3_{name}.log'
            if log.exists():
                raise FileExistsError(log)
            entry = {'name': name, 'command': command, 'gpu_before': gpu,
                     'log': str(log), 'status': 'running'}
            record['stages'].append(entry)
            REPORT.write_text(json.dumps(record, indent=2)+'\n')
            print(f'Starting {name}; log={log}', flush=True)
            before = time.monotonic()
            with log.open('x') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=remaining)
            entry.update(returncode=result.returncode, seconds=time.monotonic()-before,
                         status='complete' if result.returncode == 0 else 'failed_stop')
            if result.returncode != 0:
                raise RuntimeError(f'{name} failed: return code {result.returncode}')
            print(f'Completed {name} in {entry["seconds"]:.1f}s', flush=True)
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
