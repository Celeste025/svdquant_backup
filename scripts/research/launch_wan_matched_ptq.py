#!/usr/bin/env python3
"""E047 single-worker PTQ supervisor. Parent starts this inside named tmux."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'research_state/06_experiments/E047_wan_matched_calibration_manifest.json'
RUNNER = ROOT / 'scripts/research/ptq_wan_matched_calibration.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=MANIFEST)
    p.add_argument('--gpu', type=int, default=4, choices=range(6))
    p.add_argument('--deadline-unix', type=float)
    args = p.parse_args()
    m = json.loads(args.manifest.read_text()); rd = Path(m['report_dir'])
    output = rd / 'ptq_launcher.json'; result = rd / 'ptq_run.json'
    log = ROOT / 'results/logs/E047_ptq.log'
    assert os.environ.get('TMUX'), 'Run the supervisor in named tmux'
    assert not any(x.exists() for x in (output, result, log)), 'Keep earlier attempts'
    check_path = rd / 'ptq_check.json'; check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and not check['environment']['cuda_available']
    expected = {row['file']: row['sha256'] for row in check['sources']}
    assert expected[str(RUNNER)] == record(RUNNER)['sha256'], 'Runner changed after CPU check'
    assert expected[str(args.manifest.absolute())] == record(args.manifest)['sha256'], 'Manifest changed after check'
    prerequisites = []
    for name in ('collect_launcher.json', 'collection_summary.json'):
        path = rd / name
        assert json.loads(path.read_text())['status'] == 'complete', f'{name} not complete'
        prerequisites.append(record(path))
    cache = Path(m['data_dir']) / 'calibration/caches'
    assert {x.name for x in cache.glob('*.pt')} == {x['filename'] for x in m['selected_caches']}
    started = time.time()
    end = min(started + 21600, args.deadline_unix or float('inf'))
    assert end > started
    report = dict(experiment='E047', status='running', start_epoch=started, deadline_epoch=end,
                  gpu=args.gpu, manifest=record(args.manifest), runner=record(RUNNER),
                  launcher=record(Path(__file__)), cpu_check=record(check_path), collection=prerequisites)
    proc = None
    try:
        report['gpu_before'] = devices()[args.gpu]
        assert report['gpu_before']['memory_used_mib'] < 256 and report['gpu_before']['utilization_pct'] == 0, 'GPU busy'
        env = environment(m['python'], args.gpu)
        env['DEEPCOMPRESSOR_WAN_GATED'] = '0'
        cmd = [m['python'], '-u', str(RUNNER), '--phase', 'run', '--manifest', str(args.manifest),
               '--output', str(result), '--deadline-unix', str(end)]
        report.update(command=cmd, log=str(log), result=str(result))
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            report['pid'] = proc.pid; save(output, report)
            print('E047 PTQ started', proc.pid, 'GPU', args.gpu, flush=True)
            while proc.poll() is None:
                if time.time() >= end:
                    raise TimeoutError('E047 PTQ 21600-second process-group deadline')
                time.sleep(2)
        report['returncode'] = proc.returncode
        assert proc.returncode == 0, 'PTQ worker failed; preserve log and partial artifacts'
        assert json.loads(result.read_text())['status'] == 'complete'
        report.update(status='complete', result=str(result), result_sha256=record(result)['sha256'],
                      checkpoint_identity=record(rd / 'checkpoint_identity.json'))
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        if proc is not None:
            stop(proc); report['returncode'] = proc.returncode
        report['seconds'] = time.time()-started
        report['gpu_after'] = devices().get(args.gpu)
        save(output, report); print(report['status'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
