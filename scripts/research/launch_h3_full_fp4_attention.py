#!/usr/bin/env python3
"""E016 serial full-attention baseline with one fixed 30-minute deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from launch_h3_plain_baseline import ROOT, sha, complete
from resume_h3_plain_baseline import idle_stable

RD = ROOT/'results/research/E016'
FREEZE = RD/'frozen_contract.json'
OUTPUT = RD/'launcher.json'
MANIFEST = ROOT/'research_state/06_experiments/E016_h3_full_attention_manifest.json'
RUNNER = ROOT/'scripts/research/probe_h3_full_fp4_attention.py'
CHECK = RD/'E016_check_bf16_original.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    frozen = complete(FREEZE)
    for path, digest in frozen['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen input/source changed: {path}')
    checked = complete(CHECK)
    if checked.get('cuda_initialized') is not False:
        raise RuntimeError('Expected completed CPU-only input/source check')
    manifest = json.loads(MANIFEST.read_text())
    python = manifest['python']
    start = time.time()
    deadline = start+manifest['budget']['wall_seconds']
    report = dict(experiment='E016', status='running', start_epoch=start,
        deadline_epoch=deadline, wall_budget_seconds=1800, max_complete_dit_calls=30,
        planned_dit_calls=27, freeze_sha256=sha(FREEZE), stages=[])
    stages = [('evaluate', arm, 3) for arm in manifest['arms']]
    stages += [('bench', arm, 5) for arm in manifest['benchmark_arms']]
    try:
        for phase, arm, calls in stages:
            gpu = idle_stable(5)
            available = deadline-time.time()
            if available <= 0:
                raise TimeoutError('Original E016 wall deadline expired')
            result_path = RD/f'E016_{phase}_{arm}.json'
            if result_path.exists():
                raise FileExistsError(result_path)
            command = [python, '-u', str(RUNNER), '--phase', phase, '--arm', arm,
                       '--check-report', str(CHECK), '--deadline-unix', str(deadline)]
            log = ROOT/'results/logs'/f'E016_{phase}_{arm}.log'
            env = dict(os.environ, **manifest['environment'], CUDA_VISIBLE_DEVICES='5', PYTHONUNBUFFERED='1')
            env['PATH'] = str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH', '')
            # All compilation artifacts remain on the data disk. These do not
            # change model arithmetic; evaluation warms shapes before timing.
            env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
            env['CUDA_CACHE_PATH'] = '/data1/models/svdquant-wjq/research/cache/cuda'
            Path(env['CUDA_CACHE_PATH']).mkdir(parents=True, exist_ok=True)
            entry = dict(name=phase+'_'+arm, status='running', command=command, log=str(log),
                gpu_before=gpu, planned_dit_calls=calls, start_epoch=time.time(),
                environment={key: env[key] for key in [*manifest['environment'], 'CUDA_VISIBLE_DEVICES', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH']})
            report['stages'].append(entry)
            OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
            print(f'Start {phase} {arm} GPU5; {log}', flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = proc.pid
                OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
                try:
                    code = proc.wait(timeout=available)
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    raise
            entry.update(returncode=code, end_epoch=time.time())
            entry['seconds'] = entry['end_epoch']-entry['start_epoch']
            if code:
                raise RuntimeError(f'{phase} {arm} failed: returncode={code}')
            result = complete(result_path)
            if result['complete_dit_calls'] != calls:
                raise RuntimeError(f'{phase} {arm}: unexpected DiT call count')
            entry.update(status='complete', report=str(result_path), report_sha256=sha(result_path))
            print(f'Complete {phase} {arm} in {entry["seconds"]:.1f}s', flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        if report['stages'] and report['stages'][-1]['status'] == 'running':
            report['stages'][-1]['status'] = 'failed_stop'
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.time()-start
        OUTPUT.write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
