#!/usr/bin/env python3
"""E050: four bounded matched-calibration workers; preserve failures, no retry."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E050_wan14b_matched_calibration_manifest.json'
RUNNER = ROOT/'scripts/research/collect_wan14b_matched_calibration.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    rd = Path(m['report_dir']); rd.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'results/logs'; logs.mkdir(parents=True, exist_ok=True)
    output = rd/'collect_launcher.json'
    assert not output.exists(), 'Preserve prior launcher'
    check_path = rd/'collect_check.json'; check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and check['check_only'] and not check['cuda_initialized']
    runner_record = record(RUNNER)
    assert check['experiment'] == 'E050' and check['sources'][0] == runner_record, 'Runner changed after CPU check'
    assert check['manifest'] == record(args.manifest), 'Manifest changed after CPU check'
    start = time.time(); deadline = start + m['budget']['launcher_seconds']
    report = dict(experiment='E050', status='running', start_epoch=start, deadline_epoch=deadline,
                  manifest=record(args.manifest), runner=runner_record, launcher=record(Path(__file__)),
                  cpu_check=record(check_path), workers=[])
    owned = []; streams = []
    try:
        idle_deadline = min(deadline, time.time()+m['budget']['launch_idle_wait_seconds'])
        report['gpu_before'] = devices(); save(output, report)
        while True:
            first = devices()
            idle = all(first[g['gpu']]['memory_used_mib'] < 256 and
                       first[g['gpu']]['utilization_pct'] == 0 for g in m['worker_groups'])
            if idle:
                time.sleep(2)
                second = devices()
                if all(second[g['gpu']]['memory_used_mib'] < 256 and
                       second[g['gpu']]['utilization_pct'] == 0 for g in m['worker_groups']):
                    report['gpu_idle_observations'] = [first,second]; break
            if time.time() >= idle_deadline:
                raise TimeoutError('E050 idle GPU wait expired; no foreign process is stopped')
            time.sleep(2)
        for group in m['worker_groups']:
            gid = str(group['group']); gpu = group['gpu']
            current = devices()[gpu]
            assert current['memory_used_mib'] < 256 and current['utilization_pct'] == 0, f'GPU {gpu} acquired by another job'
            result = rd/f'collect_worker_{gid}.json'; log = logs/f'E050_collect_group{gid}.log'
            assert not result.exists() and not log.exists(), 'Keep existing worker artifacts'
            end = min(deadline, time.time()+m['budget']['worker_seconds'])
            cmd = [m['python'], '-u', str(RUNNER), '--manifest', str(args.manifest), '--group', gid,
                   '--output', str(result), '--deadline-unix', str(end)]
            stream = log.open('x'); streams.append(stream)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(m['python'], gpu), stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            item = dict(**group, pid=proc.pid, status='running', command=cmd, log=str(log),
                        output=str(result), deadline_epoch=end, gpu_at_start=current)
            owned.append((proc, item)); report['workers'].append(item)
            save(output, report); print(f'Launched group {gid} GPU{gpu}, PID {proc.pid}', flush=True)
        while True:
            done = True
            for proc, item in owned:
                code = proc.poll()
                if code is None:
                    done = False
                    if time.time() > item['deadline_epoch']:
                        raise TimeoutError(f"Worker group {item['group']} reached deadline")
                elif item['status'] == 'running':
                    item['returncode'] = code
                    assert code == 0, f"Worker group {item['group']} exited {code}; keep log"
                    worker = json.loads(Path(item['output']).read_text())
                    assert worker['status'] == 'complete'
                    assert {p['name'] for p in worker['prompts']} == set(item['prompt_names'])
                    item.update(status='complete', result=record(Path(item['output'])), actual_counts=worker['actual_counts'])
                    save(output, report)
            if done:
                break
            if time.time() > deadline:
                raise TimeoutError('E050 launcher deadline')
            time.sleep(1)
        workers = [json.loads(Path(item['output']).read_text()) for _, item in owned]
        counts = {key: sum(w['actual_counts'][key] for w in workers) for key in workers[0]['actual_counts']}
        for key, expected in m['expected_counts'].items():
            assert counts[key] == expected, f'Total count differs: {key}'
        assert counts['dit_sdpa'] == 112000
        caches = [c for w in workers for c in w['caches']]
        assert len(caches) == 64 and {c['filename'] for c in caches} == {c['filename'] for c in m['selected_caches']}
        report.update(status='complete', actual_counts=counts, caches=caches, gpu_after=devices())
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
        save(output, report)
        print(report['status'], report['seconds'], flush=True)
    # This is reached only after successful collection and owned worker cleanup.
    reducer = ROOT/'scripts/research/summarize_wan14b_matched_calibration.py'
    summary_job = rd/'collection_summary_launcher.json'
    summary_log = logs/'E050_collection_summary.log'
    assert not summary_job.exists() and not summary_log.exists(), 'Preserve summary artifacts'
    began = time.time()
    command = [m['python'], '-u', str(reducer), '--manifest', str(args.manifest), '--report-dir', str(rd)]
    summary_report = dict(experiment='E050', status='running', source=record(reducer),
        command=command, collection=record(output), start_epoch=began, timeout_seconds=180)
    save(summary_job, summary_report)
    try:
        with summary_log.open('x') as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment(m['python'], ''),
                stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        summary_report['returncode'] = completed.returncode
        assert completed.returncode == 0, 'Independent summary failed; keep collection and log'
        summary = json.loads((rd/'collection_summary.json').read_text())
        assert summary['status'] == 'complete' and summary['cuda_initialized'] is False
        summary_report.update(status='complete', result=record(rd/'collection_summary.json'))
    except BaseException:
        summary_report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        summary_report['seconds'] = time.time()-began; save(summary_job, summary_report)



if __name__ == '__main__':
    main()
