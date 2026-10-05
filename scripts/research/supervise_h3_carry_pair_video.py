#!/usr/bin/env python3
"""Run one E068 arm's two phases under the same root-assigned deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('restart', 'carry'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    rd = root/'results/research/E068'
    launch = json.loads((rd/'launch.json').read_text())
    config = launch['arms'][args.arm]
    env = os.environ.copy()
    env.update(launch['environment'])
    env['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    result = dict(arm=args.arm, supervisor_pid=os.getpid(), start_epoch=time.time(),
                  budget_start_unix=launch['budget_start_unix'], deadline_unix=launch['deadline_unix'], phases=[])
    exit_code = 1
    try:
        for phase in ('denoise', 'decode'):
            assert hashlib.sha256(Path(launch['runner']).read_bytes()).hexdigest() == launch['source_sha256']
            assert hashlib.sha256(Path(launch['plan']).read_bytes()).hexdigest() == launch['plan_sha256']
            remaining = launch['deadline_unix']-time.time()
            assert remaining > 0, 'Original deadline exhausted'
            if phase == 'decode':
                occupied = subprocess.check_output(['nvidia-smi', '-i', str(config['gpu']),
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                assert occupied == '0', f'GPU occupied between phases: {occupied} MiB'
            command = ['timeout', '--kill-after=15s', str(remaining), *config['commands'][phase]]
            entry = dict(phase=phase, command=command, start_epoch=time.time())
            result['phases'].append(entry)
            with Path(config['logs'][phase]).open('xb') as output:
                process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
                entry['timeout_pid'] = process.pid
                (rd/f'process_started_{args.arm}.json').write_text(json.dumps(result, indent=2)+'\n')
                exit_code = process.wait()
            entry.update(exit_code=exit_code, end_epoch=time.time())
            if exit_code:
                break
            completed = json.loads((rd/f'{phase}_{args.arm}.json').read_text())
            assert completed['status'] == 'complete'
        result['exit_code'] = exit_code
    except BaseException as exc:
        result.update(exit_code=1, supervisor_error=repr(exc))
        raise
    finally:
        result['end_epoch'] = time.time()
        with (rd/f'process_exit_{args.arm}.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
