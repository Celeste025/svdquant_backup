#!/usr/bin/env python3
"""Run the fixed E037 two-rank experiment, with one shared 900s deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from decode_fastwan_full_vae_control import ROOT, record, save
from resume_h3_plain_baseline import idle_stable

RD = ROOT / 'results/research/E037'


def main():
    output = RD / 'launcher.json'
    log = ROOT / 'results/logs/E037_sequence_parallel.log'
    results = [RD / f'run_rank{rank}.json' for rank in range(2)]
    assert not any(p.exists() for p in [output, log, *results]), 'Preserve previous attempts'
    check_path = RD / 'check.json'
    check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and check['cuda_initialized'] is False
    script = ROOT / 'scripts/research/bench_h3_sequence_parallel_global.py'
    prior = json.loads((ROOT / 'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python = prior['python']
    state = dict(experiment='E037', status='starting', source=record(script),
                 launcher=record(__file__), cpu_check=record(check_path),
                 physical_gpus=[0, 1], gpu_before={str(gpu): idle_stable(gpu) for gpu in (0, 1)},
                 topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True), log=str(log))
    start = time.time()
    deadline = start + 900
    command = [python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=2', str(script), '--phase', 'run', '--deadline-unix', str(deadline)]
    env = dict(os.environ, **prior['environment'], CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1', NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,GRAPH')
    cache = Path('/data1/models/svdquant-wjq/research/cache/e037')
    for key, sub in dict(XDG_CACHE_HOME='.', HF_HOME='huggingface', TORCH_HOME='torch',
                         CUDA_CACHE_PATH='cuda', TMPDIR='tmp', TRITON_CACHE_DIR='triton').items():
        path = cache / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
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
                code = worker.wait(timeout=max(.01, deadline-time.time()))
            except BaseException:
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                        worker.wait()
                raise
        state['returncode'] = code
        assert code == 0, f'E037 torchrun exited {code}; log retained'
        for result in results:
            value = json.loads(result.read_text())
            assert value['status'] == 'complete', result
        state.update(status='complete', results=[record(p) for p in results])
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
