#!/usr/bin/env python3
"""One immutable E072a launch, external hard deadline, durable process receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    rd = root/'results/research/E072a'
    launch_path = rd/'launch.json'
    launch = json.loads(launch_path.read_text())
    result = dict(experiment='E072a', supervisor_pid=os.getpid(),
                  launch_sha256=digest(launch_path), start_epoch=time.time(),
                  budget_start_unix=launch['budget_start_unix'], deadline_unix=launch['deadline_unix'])
    code = 1
    process = None
    try:
        assert os.environ.get('TMUX'), 'Named tmux required'
        for name in ('runner', 'plan', 'cpu_check', 'supervisor'):
            assert digest(launch[name]['file']) == launch[name]['sha256'], f'{name} changed after launch freeze'
        assert json.loads(Path(launch['cpu_check']['file']).read_text())['status'] == 'complete'
        remaining = launch['deadline_unix'] - time.time()
        assert remaining > 0, 'Original deadline exhausted'
        gpu_state = subprocess.check_output(['nvidia-smi', '-i', str(launch['gpu']),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        assert [int(s.strip()) for s in gpu_state.split(',')] == [0, 0], f'GPU is occupied: {gpu_state}'
        env = dict(os.environ, **launch['environment'])
        assert env['CUDA_VISIBLE_DEVICES'] == str(launch['gpu'])
        command = ['timeout', '--kill-after=15s', str(remaining), *launch['command']]
        result.update(command=command, gpu=launch['gpu'], gpu_before=gpu_state)
        with Path(launch['log']).open('xb') as stream:
            process = subprocess.Popen(command, env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            result['timeout_pid'] = process.pid
            with (rd/'process_started.json').open('x') as out:
                json.dump(result, out, indent=2)
                out.write('\n')
            result['sampled_peak_gpu_used_mib'] = 0
            while process.poll() is None:
                used = int(subprocess.check_output(['nvidia-smi', '-i', str(launch['gpu']),
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                result['sampled_peak_gpu_used_mib'] = max(result['sampled_peak_gpu_used_mib'], used)
                if used > 30*1024:
                    result['memory_stop'] = dict(gpu_used_mib=used, unix=time.time(),
                        note='Conservative total-device observation; only this owned process group is stopped.')
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(2)
            code = process.wait()
        result['worker_exit_code'] = code
        if code == 0:
            completed = json.loads(Path(launch['output']).read_text())
            assert completed['status'] == 'complete', 'Worker exit0 without complete result'
            result['completed_output_sha256'] = digest(launch['output'])
    except BaseException as exc:
        result['supervisor_error'] = repr(exc)
        code = 1
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            result['cleanup_worker_exit_code'] = process.poll()
        raise
    finally:
        result.update(exit_code=code, end_epoch=time.time(),
                      timeout_exit=code in (124, 137))
        with (rd/'process_exit.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
