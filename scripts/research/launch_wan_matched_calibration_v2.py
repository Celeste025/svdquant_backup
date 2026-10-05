#!/usr/bin/env python3
"""E047: four bounded matched-calibration groups on three GPU queues; preserve failures, no retry."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E047_wan_matched_calibration_manifest.json'
RUNNER = ROOT/'scripts/research/collect_wan_matched_calibration.py'


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
    assert any(s == runner_record for s in check['sources']), 'Runner changed after CPU check'
    assert check['manifest'] == record(args.manifest), 'Manifest changed after CPU check'
    start = time.time(); deadline = start + m['budget']['launcher_seconds']
    report = dict(experiment='E047', status='running', start_epoch=start, deadline_epoch=deadline,
                  operational_amendment='GPUs2-4 acquired unrelated contexts before launch; original four data groups queue on idle GPUs0,1,5; science and worker unchanged.',
                  manifest=record(args.manifest), runner=runner_record, launcher=record(Path(__file__)),
                  cpu_check=record(check_path), workers=[])
    owned = []; streams = []
    try:
        report['gpu_before'] = devices()
        pending = list(m['worker_groups']); running = {}
        eligible = [0, 1, 5]
        report['eligible_gpus'] = eligible
        while pending or running:
            if time.time() > deadline:
                raise TimeoutError('E047 launcher deadline')
            for gpu, (proc, item) in list(running.items()):
                code = proc.poll()
                if code is None:
                    if time.time() > item['deadline_epoch']:
                        raise TimeoutError(f"Worker group {item['group']} reached deadline")
                    continue
                item['returncode'] = code
                assert code == 0, f"Worker group {item['group']} exited {code}; keep log"
                worker = json.loads(Path(item['output']).read_text())
                assert worker['status'] == 'complete'
                assert {p['name'] for p in worker['prompts']} == set(item['prompt_names'])
                item.update(status='complete', result=record(Path(item['output'])), actual_counts=worker['actual_counts'])
                del running[gpu]; save(output, report)
            for gpu in eligible:
                if not pending or gpu in running:
                    continue
                state = devices()[gpu]
                if state['memory_used_mib'] >= 256 or state['utilization_pct'] != 0:
                    continue
                group = pending.pop(0); gid = str(group['group'])
                result = rd/f'collect_worker_{gid}.json'; log = logs/f'E047_collect_group{gid}.log'
                assert not result.exists() and not log.exists(), 'Keep existing worker artifacts'
                end = min(deadline, time.time()+m['budget']['worker_seconds'])
                cmd = [m['python'], '-u', str(RUNNER), '--manifest', str(args.manifest), '--group', gid,
                       '--output', str(result), '--deadline-unix', str(end)]
                stream = log.open('x'); streams.append(stream)
                proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(m['python'], gpu), stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(**{k:v for k,v in group.items() if k != 'gpu'}, planned_gpu=group['gpu'],gpu=gpu,
                            pid=proc.pid,status='running',command=cmd,log=str(log),output=str(result),deadline_epoch=end,
                            gpu_at_start=state)
                owned.append((proc,item));report['workers'].append(item);running[gpu]=(proc,item)
                save(output,report);print(f'Launched group {gid} GPU{gpu}, PID {proc.pid}',flush=True)
            time.sleep(1)
        workers = [json.loads(Path(item['output']).read_text()) for _, item in owned]
        counts = {key: sum(w['actual_counts'][key] for w in workers) for key in workers[0]['actual_counts']}
        for key, expected in m['expected_counts'].items():
            assert counts[key] == expected, f'Total count differs: {key}'
        assert counts['dit_sdpa'] == 84000
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


if __name__ == '__main__':
    main()
