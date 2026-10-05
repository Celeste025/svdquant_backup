#!/usr/bin/env python3
"""Supervise the fixed E040 two-GEMM information-source control on GPU0 for at most 300 seconds."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from decode_fastwan_full_vae_control import ROOT, record, save

RD = ROOT / "results/research/E040"
ARTIFACTS = Path("/data1/models/svdquant-wjq/research/20261003/E040")
from resume_h3_plain_baseline import idle_stable


def main():
    assert os.environ.get('TMUX'), 'Run inside a named tmux session'
    output = RD / 'launcher.json'
    result = RD / 'run.json'
    log = ROOT / 'results/logs/E040_side_information.log'
    assert not any(p.exists() for p in (output, result, log, ARTIFACTS)), 'Preserve existing run'
    script = ROOT / 'scripts/research/probe_h3_projection_side_information.py'
    check_path=RD/'check.json'
    check=json.loads(check_path.read_text())
    assert check['status']=='complete' and check['cuda_initialized'] is False
    for row in check['sources'].values():
        assert record(row['file'])==row, f'Checked source changed: {row["file"]}'
    prior = json.loads((ROOT / 'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = prior['python']
    state = dict(experiment='E040', status='starting', source=record(script), launcher=record(__file__),
                 gpu=0, gpu_before=idle_stable(0), cpu_check=record(check_path), log=str(log))
    start = time.time()
    deadline = start + 300
    command = [python, '-u', str(script), '--output', str(result), '--phase', 'run',
               '--deadline-unix', str(deadline)]
    env = dict(os.environ, **prior['environment'], CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    cache = Path('/data1/models/svdquant-wjq/research/20261003/E040_cache')
    for key, sub in dict(XDG_CACHE_HOME='.', HF_HOME='huggingface', TORCH_HOME='torch',
                         CUDA_CACHE_PATH='cuda', TMPDIR='tmp').items():
        path = cache / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env['PATH'] = str(Path(python).parent) + ':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:' + env.get('PATH', '')
    env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
    state.update(start_epoch=start, deadline_epoch=deadline, command=command)
    worker = None
    def interrupted(signum, frame):
        raise RuntimeError(f'Launcher interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
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
        assert code == 0, f'E040 worker exited {code}; retained log {log}'
        value = json.loads(result.read_text())
        assert value['status']=='complete' and value['actual_counts']==dict(pack=2,native_gemm=2,packet_decode=2,lr_down=2,lr_up=6,communication=0,attention=0,dit=0)
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
