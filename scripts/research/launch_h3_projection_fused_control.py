#!/usr/bin/env python3
"""One bounded E054 two-GPU strong-control run; never retries automatically."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from decode_fastwan_full_vae_control import ROOT, record, save
from resume_h3_plain_baseline import idle_stable

RD = ROOT / 'results/research/E054'
DATA = Path('/data1/models/svdquant-wjq/research/20261004/E054')
SCRIPT = ROOT / 'scripts/research/bench_h3_projection_fused_control.py'
PLAN = ROOT / 'research_state/06_experiments/E054_h3_fused_boundary_plan.md'


def main():
    assert os.environ.get('TMUX'), 'Named tmux session required'
    output = RD / 'launcher.json'
    log = ROOT / 'results/logs/E054_fused_boundary.log'
    results = [RD / f'run_rank{r}.json' for r in range(2)]
    assert not any(p.exists() for p in [output, log, DATA, *results]), 'Preserve previous attempt'
    check_path = RD / 'check.json'
    check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and check['cuda_initialized'] is False
    for item in check['sources'].values():
        assert record(item['file']) == item, f'Checked source changed: {item["file"]}'
    old = json.loads((ROOT / 'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = old['python']
    env = dict(os.environ, **old['environment'], CUDA_VISIBLE_DEVICES='0,1',
               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
               NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH')
    cache = Path('/data1/models/svdquant-wjq/research/cache/e054')
    for key, sub in {'XDG_CACHE_HOME': '.', 'CUDA_CACHE_PATH': 'cuda',
                     'TMPDIR': 'tmp', 'TRITON_CACHE_DIR': 'triton',
                     'TORCHINDUCTOR_CACHE_DIR': 'inductor'}.items():
        p = cache / sub
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p)
    env['PATH'] = str(Path(python).parent) + ':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:' + env.get('PATH', '')
    state = dict(experiment='E054', status='starting', source=record(SCRIPT),
                 launcher=record(__file__), plan=record(PLAN), cpu_check=record(check_path),
                 physical_gpus=[0, 1], wall_budget_seconds=1200, log=str(log),
                 gpu_before={str(g): idle_stable(g) for g in (0, 1)})
    started = time.time()
    deadline = started + 1200
    command = [python, '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc_per_node=2', str(SCRIPT), '--phase', 'run',
               '--plan', str(PLAN), '--data', str(DATA), '--deadline-unix', str(deadline)]
    state.update(start_epoch=started, deadline_epoch=deadline, command=command)
    worker = None
    def interrupted(signum, frame):
        raise RuntimeError(f'Supervisor received signal {signum}')
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
        assert code == 0, f'E054 torchrun returned {code}; see preserved log'
        for p in results:
            value = json.loads(p.read_text())
            assert value['experiment'] == 'E054' and value['status'] == 'complete'
            assert value['actual_dit_calls'] == 0
        state.update(status='complete', results=[record(p) for p in results])
    except BaseException:
        state.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        if worker is not None:
            state['returncode'] = worker.poll()
        state['seconds'] = time.time()-started
        save(output, state)
        print(json.dumps({'status': state['status'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
