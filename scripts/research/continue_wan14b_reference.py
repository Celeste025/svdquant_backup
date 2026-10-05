#!/usr/bin/env python3
"""E049 one-shot continuation after pinned Wan14B assets: check, generate, evaluate, summarize.

No retries or scientific policy changes. Existing GPU launchers own their
deadlines and worker process groups; this wrapper never SIGKILLs those parents.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, record, save, stop
from launch_vanilla_wan_native_v2 import environment
from launch_vanilla_wan_evaluation import base_env, PYTHON as MJ_PYTHON

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'research_state/06_experiments/E049_wan14b_reference_manifest.json'
SCRIPTS = ROOT / 'scripts/research'


def load(path):
    return json.loads(Path(path).read_text())


def process_identity(pid):
    """Identify the original process despite possible PID reuse; zombies exited."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else (pid, fields[19])
    except FileNotFoundError:
        return None


def wait_assets(m, report, output):
    run_path, launch_path = Path(m['asset_download']), Path(m['asset_supervisor'])
    original = load(launch_path)
    worker, parent = original['worker_pid'], original['supervisor_pid']
    identities = {pid: process_identity(pid) for pid in (worker, parent)}
    end = original['deadline_epoch'] + 120
    report['asset_wait'] = dict(worker_pid=worker, supervisor_pid=parent,
        process_identities=identities, deadline_epoch=end)
    save(output, report)
    while True:
        launch, run = load(launch_path), load(run_path)
        assert launch['worker_pid'] == run['worker_pid'] == worker, 'Asset worker changed; do not follow another attempt'
        for value in (launch, run):
            assert value['status'] in ('running', 'complete'), 'Asset preparation failed; preserve evidence and stop'
        exited = all(process_identity(pid) is None or
                     (identities[pid] is not None and process_identity(pid) != identities[pid])
                     for pid in identities)
        if launch['status'] == run['status'] == 'complete' and exited:
            assert launch['returncode'] == 0
            assert run['revision'] == m['model_revision'] and run['repo'] == m['model_repo']
            assert run['header_validation']['status'] == 'complete' and run['header_validation']['index_exact']
            report['asset_wait'].update(status='complete', original_processes_exited=True,
                                       download=record(run_path), supervisor=record(launch_path))
            save(output, report); return
        if exited and (launch['status'] != 'complete' or run['status'] != 'complete'):
            raise RuntimeError('Original asset processes exited without complete receipts; no restart')
        if time.time() >= end:
            raise TimeoutError('Asset completion wait exceeded original deadline + 120s')
        time.sleep(min(15, max(.1, end-time.time())))


def wait_evaluation_gpus(report, output):
    started = time.time(); end = started + 1800
    report['evaluation_gpu_wait'] = dict(start_epoch=started, deadline_epoch=end)
    save(output, report)
    while True:
        state = devices()
        if all(state[g]['memory_used_mib'] < 256 and state[g]['utilization_pct'] == 0 for g in (0, 1)):
            report['evaluation_gpu_wait'].update(status='complete', seconds=time.time()-started,
                                                  gpu_at_ready={g: state[g] for g in (0, 1)})
            save(output, report); return
        if time.time() >= end:
            raise TimeoutError('GPU0/1 unavailable for E049 evaluation after 1800s; no processes killed')
        time.sleep(min(30, max(.1, end-time.time())))


def execute(stage, env, report, output):
    log = ROOT / f"results/logs/E049_chain_{stage['name']}.log"
    assert not log.exists(), 'Keep previous stage logs'
    assert not stage['result'].exists(), f"Keep previous result: {stage['result']}"
    item = dict(name=stage['name'], command=stage['command'], log=str(log), status='running',
                start_epoch=time.time(), cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
                timeout_seconds=stage['timeout'])
    report['stages'].append(item); save(output, report)
    proc = None
    try:
        with log.open('x') as stream:
            proc = subprocess.Popen(stage['command'], cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            item['pid'] = proc.pid; save(output, report)
            # GPU parents retain their own published budgets and finally cleanup.
            proc.wait(timeout=stage['timeout'])
        item['returncode'] = proc.returncode
        assert proc.returncode == 0, f"{stage['name']} failed; see {log}"
        assert load(stage['result'])['status'] == 'complete', 'Stage report incomplete'
        item.update(status='complete', result=record(stage['result']))
    finally:
        if proc is not None and proc.poll() is None:
            if stage['timeout'] is None:
                # SIGINT raises KeyboardInterrupt inside the existing launcher,
                # whose finally stops its separately-sessioned worker groups.
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    item['cleanup'] = 'launcher still alive; not forcibly killed; inspect recorded PID'
                    save(output, report)
                    raise RuntimeError('GPU launcher cleanup did not finish; no subsequent stage launched')
            else:
                stop(proc)  # CPU-only child has no GPU worker tree.
        if proc is not None:
            item['returncode'] = proc.returncode
        item['seconds'] = time.time()-item['start_epoch']
        if item['status'] == 'running':
            item['status'] = 'failed_preserved'
        save(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); m = load(MANIFEST); rd = Path(m['report_dir'])
    ids = list(dict.fromkeys(str(c['prompt_id']) for c in m['cases']))
    stages = [
        dict(name='reference_check', command=[m['python'], '-u', str(SCRIPTS/'run_wan14b_reference.py'),
             '--manifest', str(MANIFEST), '--prompt-ids', *ids, '--check-only'],
             result=rd/'check_bf16.json', timeout=300, env='native_cpu'),
        dict(name='generation', command=[m['python'], '-u', str(SCRIPTS/'launch_wan14b_reference.py')],
             result=rd/'launcher.json', timeout=None, env='native_cpu'),
        dict(name='evaluation', command=[MJ_PYTHON, '-u', str(SCRIPTS/'launch_wan14b_reference_evaluation.py')],
             result=rd/'evaluation_launcher.json', timeout=None, env='mj_cpu'),
        dict(name='summary', command=[MJ_PYTHON, '-u', str(SCRIPTS/'summarize_wan14b_reference.py')],
             result=rd/'evaluation_summary.json', timeout=180, env='mj_cpu')]
    for stage in stages:
        assert Path(stage['command'][0]).is_file() and Path(stage['command'][2]).is_file()
    check = load(rd/'evaluation_check.json')
    assert check['status'] == 'complete' and not check['cuda_initialized']
    if args.check_only:
        print(json.dumps(dict(status='ready_to_wait', model_calls=0, subprocesses_launched=0,
              asset_wait='download + supervisor complete AND original processes exited; deadline + 120s',
              evaluation_gpu_wait_seconds=1800, stages=stages), default=str, indent=2)); return
    assert os.environ.get('TMUX'), 'Start the continuation in a named tmux session'
    output = rd/'chain_launcher.json'; assert not output.exists(), 'Keep previous continuation receipt'
    for stage in stages:
        assert not stage['result'].exists(), 'No automatic resume/overwrite of existing E049 stages'
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    report = dict(experiment='E049', status='running', start_epoch=time.time(), source=record(Path(__file__)),
                  manifest=record(MANIFEST), stages=[])
    try:
        wait_assets(m, report, output)
        native_env = environment(m['python'], ''); native_env['CUDA_VISIBLE_DEVICES'] = ''
        mj_env = base_env(); mj_env['CUDA_VISIBLE_DEVICES'] = ''
        for stage in stages:
            if stage['name'] == 'evaluation':
                wait_evaluation_gpus(report, output)
            execute(stage, native_env if stage['env'] == 'native_cpu' else mj_env, report, output)
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc()); raise
    finally:
        report['seconds'] = time.time()-report['start_epoch']; save(output, report)
        print(report['status'], str(output), flush=True)


if __name__ == '__main__':
    main()
