#!/usr/bin/env python3
"""Supervise the fixed E026 two-attention intervention on GPU1 for at most 600 seconds."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from decode_fastwan_full_vae_control import ROOT, record, save

RD = ROOT / "results/research/E026"
ARTIFACTS = Path("/data1/models/svdquant-wjq/research/20261003/E026")
from resume_h3_plain_baseline import idle_stable


def main():
    output = RD / 'launcher.json'
    result = RD / 'run.json'
    log = ROOT / 'results/logs/E026_qmean.log'
    assert not any(p.exists() for p in (output, result, log, ARTIFACTS)), 'Preserve existing run'
    script = ROOT / 'scripts/research/probe_h3_query_mean_k4.py'
    prior = json.loads((ROOT / 'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = prior['python']
    state = dict(experiment='E026', status='starting', source=record(script), launcher=record(__file__),
                 gpu=1, gpu_before=idle_stable(1), log=str(log))
    start = time.time()
    deadline = start + 600
    command = [python, '-u', str(script), '--output', str(result), '--phase', 'run',
               '--deadline-unix', str(deadline)]
    env = dict(os.environ, **prior['environment'], CUDA_VISIBLE_DEVICES='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='6', PYTHONUNBUFFERED='1')
    cache = Path('/data1/models/svdquant-wjq/research/20261003/E026_cache')
    for key, sub in dict(XDG_CACHE_HOME='.', HF_HOME='huggingface', TORCH_HOME='torch',
                         CUDA_CACHE_PATH='cuda', TMPDIR='tmp').items():
        path = cache / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PATH'] = str(Path(python).parent) + ':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:' + env.get('PATH', '')
    env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
    state.update(start_epoch=start, deadline_epoch=deadline, command=command)
    worker = None
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', pid=worker.pid)
            save(output, state)
            try:
                code = worker.wait(timeout=max(.01, deadline-time.time()))
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
        assert code == 0, f'Attention worker exited {code}; retained log {log}'
        value = json.loads(result.read_text())
        assert value['status'] == 'complete' and value['complete_dit_calls'] == 0
        assert value['attention_calls'] == 2
        state.update(status='complete', result=str(result), result_sha256=record(result)['sha256'])
    except BaseException:
        state.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        if worker is not None:
            state['returncode'] = worker.poll()
        state['seconds'] = time.time()-start
        save(output, state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
