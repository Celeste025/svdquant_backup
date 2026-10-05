#!/usr/bin/env python3
"""E048: four prompt workers on three GPU queues; new checkpoint only, no retries."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E048_matched_wan_native_manifest.json'
RUNNER = ROOT/'scripts/research/run_matched_wan_native.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    rd = Path(m['report_dir']); rd.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'results/logs'; logs.mkdir(parents=True, exist_ok=True)
    output = rd/'launcher.json'
    assert not output.exists(), 'Preserve previous launcher'
    ptq_path = Path(m['ptq_run']); ptq = json.loads(ptq_path.read_text())
    assert ptq['status'] == 'complete', 'E047 PTQ is not complete'
    check_path = rd/'check_svdquant_nvfp4.json'; check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and not check['cuda_available'] and not check['cuda_initialized']
    assert check['selected_cases'] == m['cases'], 'Require one CPU check covering all eight fixed cases'
    assert check['manifest'] == record(args.manifest) and check['matched_ptq_run'] == record(ptq_path)
    assert record(RUNNER) in check['sources'], 'Worker changed after CPU check'
    start = time.time(); deadline = start + m['budget']['launcher_timeout_seconds']
    report = dict(experiment='E048', status='running', start_epoch=start, deadline_epoch=deadline,
                  manifest=record(args.manifest), runner=record(RUNNER), launcher=record(Path(__file__)),
                  cpu_check=record(check_path), matched_ptq_run=record(ptq_path), workers=[], pending=list(m['worker_groups']))
    owned = []; streams = []; running = {}
    try:
        report['gpu_before'] = devices(); save(output, report)
        while report['pending'] or running:
            for gpu, (proc, item, result) in list(running.items()):
                code = proc.poll()
                if code is None:
                    if time.time() > item['deadline_epoch']:
                        raise TimeoutError(f'Worker {proc.pid} reached deadline')
                    continue
                item['returncode'] = code
                assert code == 0, f'Worker {proc.pid} exited {code}; retain log'
                value = json.loads(result.read_text())
                assert value['status'] == 'complete' and value['experiment'] == 'E048'
                assert len(value['cases']) == 2*len(item['prompt_ids'])
                item.update(status='complete', result=record(result), actual_counts=value['actual_counts'])
                del running[gpu]; save(output, report)
            if time.time() > deadline:
                raise TimeoutError('E048 launcher deadline')
            for group in list(report['pending']):
                gpu = group['gpu']
                if gpu in running:
                    continue
                state = devices()[gpu]
                if state['memory_used_mib'] >= 256 or state['utilization_pct'] != 0:
                    continue
                arm = group['arm']; ids = group['prompt_ids']; key = arm+'_'+'-'.join(map(str, ids))
                result = rd/f'worker_{key}.json'; log = logs/f'E048_worker_{key}.log'
                assert not result.exists() and not log.exists(), 'Preserve worker artifacts'
                end = min(deadline, time.time()+m['budget']['worker_timeout_seconds'])
                cmd = [m['python'], '-u', str(RUNNER), '--manifest', str(args.manifest), '--arm', arm,
                       '--prompt-ids', *map(str, ids), '--output', str(result), '--deadline-unix', str(end)]
                stream = log.open('x'); streams.append(stream)
                proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(m['python'], gpu), stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(**group, pid=proc.pid, deadline_epoch=end, command=cmd, output=str(result),
                            log=str(log), status='running', gpu_at_start=state)
                report['workers'].append(item); report['pending'].remove(group)
                running[gpu] = (proc, item, result); owned.append((proc, item))
                save(output, report); print(f'Launched {key} GPU{gpu}, PID {proc.pid}', flush=True)
            if running or report['pending']:
                time.sleep(1)
        workers = [json.loads(Path(item['output']).read_text()) for _, item in owned]
        cases = [case for worker in workers for case in worker['cases']]
        assert len(cases) == 8 and {c['case_id'] for c in cases} == {c['case_id'] for c in m['cases']}
        counts = {k: sum(w['actual_counts'][k] for w in workers) for k in workers[0]['actual_counts']}
        for key, expected in m['expected_new_totals'].items():
            assert (len(cases) if key == 'videos' else counts[key]) == expected, f'Final count differs: {key}'
        report.update(status='complete', actual_counts=counts, videos=len(cases), gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        for proc, item in owned:
            stop(proc); item['returncode'] = proc.returncode
            if item['status'] == 'running':
                item['status'] = 'stopped_with_launcher'
        for stream in streams:
            stream.close()
        report['seconds'] = time.time()-start
        save(output, report); print(report['status'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
