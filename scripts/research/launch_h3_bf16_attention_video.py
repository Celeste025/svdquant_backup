#!/usr/bin/env python3
"""Bounded E083 process launcher; run one replica in its own tmux session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'scripts/research/run_h3_bf16_attention_video.py'
REPORTS = ROOT / 'results/research/E083'
PYTHONS = {
    'denoise': '/data1/models/svdquant-wjq/research/envs/nvfp4-native-20261002/bin/python',
    'decode': '/home/wjq/.venvs/minimax-h3-svdquant-recovered/bin/python',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica', type=int, choices=(0, 1), required=True)
    replica = parser.parse_args().replica
    manifest = json.loads((ROOT / 'research_state/06_experiments/E083_bf16_attention_video_manifest.json').read_text())
    for arm in manifest['arms']:
        check = json.loads((REPORTS / f'check_{arm}_r{replica}.json').read_text())
        assert check['status'] == 'complete' and check['cuda_initialized'] is False
    assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == manifest['runner']['sha256']
    gpu = manifest['budget']['generation_gpus'][str(replica)]
    memory, utilization = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert int(memory) < 200 and int(utilization) == 0, 'Assigned GPU is occupied'
    output = REPORTS / f'launcher_r{replica}.json'
    assert not output.exists(), 'Preserve prior execution'
    started = time.time()
    deadline = started + manifest['budget']['wall_seconds_per_replica']
    record = dict(status='running', replica=replica, gpu=gpu, started_unix=started,
                  deadline_unix=deadline, phases=[])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', CUDA_HOME='/usr/local/cuda', FLASHINFER_WORKSPACE_BASE='/data1/models/svdquant-wjq/research/cache/flashinfer', TMPDIR='/data1/models/svdquant-wjq/research/cache/tmp', TORCH_EXTENSIONS_DIR='/data1/models/svdquant-wjq/research/cache/torch_extensions', MAX_JOBS='4', TRITON_CACHE_DIR='/data1/models/svdquant-wjq/research/cache/triton', CUDA_CACHE_PATH='/data1/models/svdquant-wjq/research/cache/cuda')
    env['PATH']='/usr/local/cuda/bin:/home/wjq/.conda/envs/convrot-wan/bin:'+env['PATH']
    try:
        for arm in manifest['arms']:
            for phase in ('denoise', 'decode'):
                log = REPORTS / f'{phase}_{arm}_r{replica}.log'
                row = dict(arm=arm, phase=phase, started_unix=time.time(), log=str(log))
                record['phases'].append(row)
                with log.open('x') as stream:
                    command = [PYTHONS[phase], '-u', str(RUNNER), '--phase', phase,
                               '--replica', str(replica), '--arm', arm, '--deadline-unix', str(deadline)]
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    row['pid'] = child.pid
                    output.write_text(json.dumps(record, indent=2) + '\n')
                    try:
                        code = child.wait(timeout=max(0.01, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                        row['timed_out'] = True
                        raise RuntimeError('Original shared deadline exhausted')
                row.update(exit_code=code, finished_unix=time.time())
                assert code == 0, f'{phase} failed; preserve output and stop'
                result = json.loads((REPORTS / f'{phase}_{arm}_r{replica}.json').read_text())
                assert result['status'] == 'complete'
                assert sum(p.stat().st_size for p in Path(manifest['data_dir']).rglob('*')
                           if p.is_file()) < 5 * 2**30, 'Data budget exceeded'
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed_stop', error=repr(exc))
        raise
    finally:
        record['finished_unix'] = time.time()
        output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
