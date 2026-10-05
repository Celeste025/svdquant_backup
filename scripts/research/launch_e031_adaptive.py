#!/usr/bin/env python3
"""Bounded GPU0 adaptive center cost and precision intervention (E031)."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from decode_fastwan_full_vae_control import ROOT, record, save
from resume_h3_plain_baseline import idle_stable

RD = ROOT / 'results/research/E031'


def main():
    result = RD / 'run.json'
    output = RD / 'launcher.json'
    log = ROOT / 'results/logs/E031.log'
    assert not any(p.exists() for p in (result, output, log)), 'Preserve existing attempt'
    check_path = RD / 'check.json'
    check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and check['cuda_initialized'] is False
    script = ROOT / 'scripts/research/probe_h3_adaptive_centers.py'
    prior = json.loads((ROOT / 'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = prior['python']
    state = dict(experiment='E031', status='starting', source=record(script),
                 launcher=record(__file__), cpu_check=record(check_path),
                 gpu=0, gpu_before=idle_stable(0), log=str(log))
    start = time.time()
    deadline = start + 600
    command = [python, '-u', str(script), '--phase', 'run', '--output', str(result),
               '--deadline-unix', str(deadline)]
    env = dict(os.environ, **prior['environment'], CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='6',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    cache = Path('/data1/models/svdquant-wjq/research/cache/e031')
    for key, sub in dict(XDG_CACHE_HOME='.', HF_HOME='huggingface', TORCH_HOME='torch',
                         CUDA_CACHE_PATH='cuda', TMPDIR='tmp').items():
        path = cache / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['TRITON_CACHE_DIR'] = str(cache / 'triton')
    env['PATH'] = str(Path(python).parent) + ':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:' + env.get('PATH', '')
    state.update(start_epoch=start, deadline_epoch=deadline, command=command)
    worker = None
    try:
        with log.open('x') as stream:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', pid=worker.pid)
            save(output, state)
            try:
                code = worker.wait(timeout=max(.01, deadline - time.time()))
            except BaseException:
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                        worker.wait()
                raise
        state['returncode'] = code
        assert code == 0, f'E031 worker exited {code}; log retained'
        value = json.loads(result.read_text())
        assert value['status'] == 'complete'
        state.update(status='complete', result=str(result), result_sha256=record(result)['sha256'])
    except BaseException:
        state.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        if worker is not None:
            state['returncode'] = worker.poll()
        state['seconds'] = time.time() - start
        save(output, state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
