#!/usr/bin/env python3
"""Bounded, logged E022 replay supervisor; only its own child process is managed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from resume_h3_plain_baseline import idle_stable

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E084'
PYTHON = '/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python'


def main():
    RD.mkdir(parents=True, exist_ok=True)
    out = RD / 'launcher.json'
    assert not out.exists()
    r = dict(status='running', start=time.time(), gpu=0, gpu_before=idle_stable(0))
    cache = '/data1/models/svdquant-wjq/research/cache'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_HOME='/usr/local/cuda',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        SVDQUANT_DATA_ROOT='/data1/models/svdquant-wjq', OMP_NUM_THREADS='6',
        TRITON_CACHE_DIR=cache+'/triton', CUDA_CACHE_PATH=cache+'/cuda',
        TORCH_EXTENSIONS_DIR=cache+'/torch_extensions', TMPDIR=cache+'/tmp', MAX_JOBS='4')
    env['PATH'] = str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH', '')
    command = [PYTHON, '-u', str(ROOT/'scripts/research/replay_wan_qad_timesteps_e084.py')]
    r['command'] = command
    try:
        with (ROOT/'results/logs/E084_replay.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            r['pid'] = proc.pid
            out.write_text(json.dumps(r, indent=2)+'\n')
            try:
                code = proc.wait(timeout=2800)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
        r['returncode'] = code
        assert code == 0
        assert json.loads((RD/'run.json').read_text())['status'] == 'complete'
        r['status'] = 'complete'
    except BaseException:
        r.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        r['seconds'] = time.time()-r['start']
        out.write_text(json.dumps(r, indent=2)+'\n')


if __name__ == '__main__':
    main()
