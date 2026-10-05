#!/usr/bin/env python3
"""Resume E014 after a zero-memory trailing-utilization sample; original deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from launch_h3_plain_baseline import ROOT, PYTHON, RD, RUNNER, CHECK, FREEZE, sha, complete

PREVIOUS = RD/'launcher.json'
OUTPUT = RD/'launcher_resume1.json'

def idle_stable(gpu):
    samples = []
    consecutive = 0
    for _ in range(20):
        row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        uuid, memory, use = [v.strip() for v in row.strip().split(',')]
        sample = dict(index=gpu, uuid=uuid, memory_mib=int(memory), utilization_percent=int(use))
        samples.append(sample)
        if int(memory) > 64:
            raise RuntimeError(f'GPU{gpu} occupied: {sample}')
        consecutive = consecutive+1 if int(use) <= 1 else 0
        if consecutive == 2:
            return dict(**sample, samples=samples)
        time.sleep(0.5)
    raise RuntimeError(f'GPU{gpu} utilization did not settle: {samples}')

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(PREVIOUS.read_text())
    assert prior['status'] == 'failed_stop'
    assert prior['error'] == "RuntimeError('GPU5 not idle: GPU-9413777f-eb0d-b7df-260d-8e87cae80c9b, 0, 13')"
    assert [s['name'] for s in prior['stages']] == ['export', 'evaluate_bf16', 'evaluate_svd']
    assert prior['freeze_sha256'] == sha(FREEZE)
    freeze = complete(FREEZE)
    for path, digest in freeze['files'].items():
        assert sha(path) == digest, path
    for stage in prior['stages']:
        assert stage['status'] == 'complete' and stage['returncode'] == 0
        complete(stage['report'])
        assert sha(stage['report']) == stage['report_sha256']
    deadline = prior['deadline_epoch']
    record = dict(experiment='E014', status='running', start_epoch=prior['start_epoch'],
        deadline_epoch=deadline, wall_budget_seconds=2700, max_complete_dit_calls=30,
        planned_dit_calls=24, freeze_sha256=sha(FREEZE),
        continuation_source=dict(file=str(Path(__file__).resolve()), sha256=sha(__file__)),
        previous_launcher=dict(file=str(PREVIOUS), sha256=sha(PREVIOUS)),
        inherited_stage_count=3, resume_epoch=time.time(), stages=prior['stages'].copy(),
        reason='Previous process exited successfully; GPU memory zero but trailing utilization 13%. Preserve failure; require two settled samples; keep original deadline and all experimental code/data.')
    stages = [('evaluate', 'plain', 3)]+[('bench', arm, 5) for arm in ['bf16', 'svd', 'plain']]
    try:
        for phase, arm, calls in stages:
            name = phase+'_'+arm
            before_gpu = idle_stable(5)
            available = deadline-time.time()
            if available <= 0:
                raise TimeoutError('Original E014 deadline expired')
            result_path = RD/f'E014_{phase}_{arm}.json'
            if result_path.exists():
                raise FileExistsError(result_path)
            command = [PYTHON, '-u', str(RUNNER), '--phase', phase, '--arm', arm,
                '--check-report', str(CHECK), '--deadline-unix', str(deadline)]
            log = ROOT/'results/logs'/f'E014_{name}.log'
            entry = dict(name=name, status='running', command=command, log=str(log),
                gpu_before=before_gpu, planned_dit_calls=calls, start_epoch=time.time())
            record['stages'].append(entry)
            OUTPUT.write_text(json.dumps(record, indent=2)+'\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', PYTHONUNBUFFERED='1')
            env['PATH'] = str(Path(PYTHON).parent)+':'+env.get('PATH', '')
            before = time.time()
            print(f'Start {name} GPU5; {log}', flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = proc.pid
                OUTPUT.write_text(json.dumps(record, indent=2)+'\n')
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
            entry.update(returncode=code, seconds=time.time()-before, end_epoch=time.time())
            if code:
                raise RuntimeError(f'{name} failed: {code}')
            complete(result_path)
            entry.update(status='complete', report=str(result_path), report_sha256=sha(result_path))
            print(f'Complete {name} in {entry["seconds"]:.1f}s', flush=True)
        record['status'] = 'complete'
    except BaseException as exc:
        if record['stages'] and record['stages'][-1]['status'] == 'running':
            record['stages'][-1]['status'] = 'failed_stop'
        record.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['seconds_total'] = time.time()-record['start_epoch']
        OUTPUT.write_text(json.dumps(record, indent=2)+'\n')

if __name__ == '__main__':
    main()
