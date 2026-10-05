#!/usr/bin/env python3
"""E053: one bounded full Wan14B PTQ worker in named tmux; no retries or follow-ups."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'research_state/06_experiments/E053_wan14b_matched_full_ptq_manifest.json'
RUNNER = ROOT / 'scripts/research/ptq_wan14b_matched_calibration.py'


def read(path):
    return json.loads(Path(path).read_text())


def bound_file(expected):
    path = Path(expected['file']).resolve()
    actual = record(path)
    assert all(actual[k] == expected[k] for k in ('bytes', 'sha256')), f'Bound file changed: {path}'
    return path, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args(); args.manifest = args.manifest.resolve()
    m = read(args.manifest); budget = m['budget']; gpu = budget['gpu']
    assert m['experiment'] == 'E053' and gpu == 4
    assert (budget['worker_seconds'], budget['launcher_seconds'], budget['initial_gpu_wait_seconds']) == (129600, 129900, 900)
    assert os.environ.get('TMUX'), 'Start this supervisor in named tmux'
    rd = Path(m['report_dir']); rd.mkdir(parents=True, exist_ok=True)
    output, result = rd / 'launcher.json', rd / 'run.json'
    log = ROOT / 'results/logs/E053_full_ptq.log'
    run_dir, checkpoint_dir = Path(m['run_dir']), Path(m['checkpoint_dir'])
    assert not any(p.exists() for p in (output, result, log, run_dir, checkpoint_dir)), 'Preserve previous attempts'
    check_path = rd / 'check.json'; check = read(check_path)
    assert check['status'] == 'complete' and check['phase'] == 'check'
    assert check['operation'] == 'wan14b_matched_full_ptq' and check['prepared_only'] is True
    assert check['manifest_experiment'] == 'E053'
    assert check['cuda_initialized'] is False and check['model_weights_loaded'] is False
    assert check['meta_registration']['blocks'] == 40 and check['meta_registration']['target_count'] == 400
    checked_sources = {str(Path(row['file']).resolve()): row for row in check['sources']}
    for path in (RUNNER, args.manifest):
        bound_file(checked_sources[str(path)])
    runner_path, _ = bound_file(m['runner'])
    assert runner_path == RUNNER, 'Manifest must select the frozen full PTQ worker'
    resource_path, resource_record = bound_file(m['resource_result'])
    resource_launcher_path, resource_launcher_record = bound_file(m['resource_launcher'])
    resource, resource_launcher = read(resource_path), read(resource_launcher_path)
    assert resource['experiment'] == resource_launcher['experiment'] == 'E052'
    assert resource['phase'] == 'run' and resource['status'] == resource_launcher['status'] == 'complete'
    assert resource_launcher['returncode'] == 0
    assert resource_launcher['result']['sha256'] == resource_record['sha256']
    assert Path(resource_launcher['result']['file']).resolve() == resource_path
    collection_dir = Path(m['collection_dir'])
    collection_paths = [collection_dir / name for name in ('collect_launcher.json', 'collection_summary.json')]
    collection, summary = (read(path) for path in collection_paths)
    assert collection['status'] == summary['status'] == 'complete'
    assert collection['manifest'] == summary['current_manifest'] == m['collection_manifest']
    assert all(w['status'] == 'complete' and w['returncode'] == 0 for w in collection['workers'])
    assert summary['cache_count'] == 64 and summary['prompt_count'] == 14
    started = time.time(); wait_end = started + budget['initial_gpu_wait_seconds']
    report = dict(experiment='E053', status='waiting_for_gpu', start_epoch=started,
        supervisor_pid=os.getpid(), source=record(Path(__file__)), runner=record(RUNNER),
        manifest=record(args.manifest), cpu_check=record(check_path), gpu=gpu,
        resource_result=resource_record, resource_launcher=resource_launcher_record,
        collection=[record(path) for path in collection_paths],
        gpu_wait_deadline_epoch=wait_end, run_dir=str(run_dir), checkpoint_dir=str(checkpoint_dir),
        budget_boundary='Initial GPU wait is separately capped at900s; launcher/worker budgets begin only after GPU readiness.',
        scope='Full40-block PTQ, all64 records, no partial reuse, no automatic retry or subsequent generation')
    save(output, report); proc = None
    def interrupted(*unused):
        raise KeyboardInterrupt('Supervisor interrupted; stop only its owned worker process group')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        previous = None
        while True:
            if time.time() >= wait_end:
                raise TimeoutError('GPU4 did not become idle within900s; no foreign process stopped')
            first = devices()[gpu]
            if first['memory_used_mib'] < 256 and first['utilization_pct'] == 0:
                time.sleep(2)
                second = devices()[gpu]
                if second['memory_used_mib'] < 256 and second['utilization_pct'] == 0:
                    last = devices()[gpu]
                    if last['memory_used_mib'] < 256 and last['utilization_pct'] == 0:
                        assert time.time() < wait_end, 'GPU wait budget expired before spawn'
                        report['gpu_before'] = [first, second, last]
                        break
            if first != previous:
                report['last_gpu_wait'] = dict(epoch=time.time(), state=first)
                save(output, report); previous = first
            time.sleep(min(5, max(0, wait_end-time.time())))
        spawn_epoch = time.time()
        launch_end = spawn_epoch + budget['launcher_seconds']
        worker_end = min(launch_end, spawn_epoch + budget['worker_seconds'])
        command = [m['python'], '-u', str(RUNNER), '--phase', 'run', '--manifest', str(args.manifest),
            '--output', str(result), '--run-dir', str(run_dir), '--checkpoint-dir', str(checkpoint_dir),
            '--resource-report', str(resource_path), '--deadline-unix', str(worker_end)]
        env = environment(m['python'], gpu); env['DEEPCOMPRESSOR_WAN_GATED'] = '0'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            report.update(status='running', worker_pid=proc.pid, command=command, log=str(log), output=str(result),
                worker_start_epoch=spawn_epoch, worker_deadline_epoch=worker_end,
                launch_start_epoch=spawn_epoch, launcher_deadline_epoch=launch_end,
                effective_worker_seconds=worker_end-spawn_epoch, gpu_wait_seconds=spawn_epoch-started)
            save(output, report); print('E053 full PTQ started', proc.pid, 'GPU', gpu, flush=True)
            while proc.poll() is None:
                if time.time() >= worker_end:
                    raise TimeoutError('E053 full PTQ bounded worker deadline')
                time.sleep(2)
        report['returncode'] = proc.returncode
        assert proc.returncode == 0, 'Worker failed; preserve log and partial artifacts, no retry'
        value = read(result)
        assert value['status'] == 'complete' and value['phase'] == 'run' and value['experiment'] == 'E053'
        assert value['partial_artifacts_loaded'] is False
        identity_path, identity_record = bound_file(value['checkpoint_identity'])
        assert read(identity_path)['status'] == 'complete'
        report.update(status='complete', result=record(result), checkpoint_identity=identity_record)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        if proc is not None:
            stop(proc); report['returncode'] = proc.returncode
        report['seconds'] = time.time()-started
        try:
            report['gpu_after'] = devices().get(gpu)
        except Exception as error:
            report['gpu_after_error'] = repr(error)
        save(output, report); print(report['status'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
