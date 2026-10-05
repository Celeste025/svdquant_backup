#!/usr/bin/env python3
"""E051: four bounded workers on two idle-only GPU queues. No retries or foreign-process cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from launch_vanilla_wan_reference import devices, stop

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E051_wan14b_plain_native_manifest.json'
RUNNER = ROOT/'scripts/research/run_wan14b_plain_native.py'


def record(path):
    p = Path(path).resolve()
    return dict(file=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest', type=Path, default=MANIFEST)
    args = p.parse_args(); m = json.loads(args.manifest.read_text())
    assert os.environ.get('TMUX'), 'Run the launcher inside named tmux'
    rd = Path(m['report_dir']); logs = ROOT/'results/logs'; rd.mkdir(parents=True, exist_ok=True); logs.mkdir(exist_ok=True)
    output = rd/'launcher.json'; assert not output.exists(), 'Preserve previous attempt'
    check_path = rd/'check_plain.json'; check = json.loads(check_path.read_text())
    assert check['status'] == 'complete' and check['cuda_available'] is False and check['model_weights_loaded'] is False
    assert len(check['selected_cases']) == 8 and {c['case_id'] for c in check['selected_cases']} == {c['case_id'] for c in m['cases']}
    for source in check['sources']:
        assert record(source['file']) == source, f'CPU-check source changed: {source["file"]}'
    assert any(Path(s['file']).resolve() == RUNNER.resolve() for s in check['sources'])
    assert record(args.manifest) == check['manifest']
    assert record(m['asset_download']) == check['asset_download'] and record(m['asset_supervisor']) == check['asset_supervisor']
    download = json.loads(Path(m['asset_download']).read_text()); supervisor = json.loads(Path(m['asset_supervisor']).read_text())
    assert download['status'] == supervisor['status'] == 'complete' and supervisor['returncode'] == 0
    assert download['header_validation']['status'] == 'complete' and download['header_validation']['index_exact']
    started = time.time(); deadline = started+m['budget']['launcher_timeout_seconds']
    report = dict(experiment='E051', status='running', start_epoch=started, deadline_epoch=deadline,
        manifest=record(args.manifest), runner=record(RUNNER), launcher=record(Path(__file__)), cpu_check=record(check_path),
        inherited_launcher_helpers=record(ROOT/'scripts/research/launch_vanilla_wan_reference.py'), workers=[], waiting=[])
    pending = list(m['worker_groups']); owned = []; streams = []; idle = {}; previous = None
    try:
        while pending or any(p.poll() is None for p,_,_,_ in owned):
            if time.time() >= deadline:
                raise TimeoutError('Global E051 deadline')
            for proc, end, result, row in owned:
                rc = proc.poll()
                if rc is None:
                    if time.time() >= end:
                        raise TimeoutError(f'Worker deadline: {proc.pid}')
                elif row['status'] == 'running':
                    row['returncode'] = rc
                    if rc:
                        raise RuntimeError(f'Worker {proc.pid} exited {rc}; preserve failure')
                    done = json.loads(result.read_text()); assert done['status'] == 'complete'
                    assert len(done['cases']) == 2 and all(c['status'] == 'complete' for c in done['cases'])
                    row.update(status='complete', result=record(result)); save(output, report)
            if pending:
                current = devices()
                signature = {g['gpu']: current[g['gpu']] for g in pending}
                if signature != previous:
                    report['waiting'].append(dict(epoch=time.time(), devices=signature)); previous = signature
                occupied = {row['gpu'] for proc,_,_,row in owned if proc.poll() is None}
                for gpu in {group['gpu'] for group in pending}:
                    dev = current[gpu]
                    clear = gpu not in occupied and dev['memory_used_mib'] < 256 and dev['utilization_pct'] == 0
                    idle[gpu] = idle.get(gpu, 0)+1 if clear else 0
                for group in list(pending):
                    gpu, pid = group['gpu'], group['prompt_id']; dev = current[gpu]
                    if gpu in occupied or idle[gpu] < 2:
                        continue
                    # Check immediately before this particular worker is launched.
                    last = devices()[gpu]
                    if last['memory_used_mib'] >= 256 or last['utilization_pct'] != 0:
                        idle[gpu] = 0; continue
                    result = rd/f'worker_{pid}.json'; log = logs/f'E051_worker_{pid}.log'
                    assert not result.exists() and not log.exists(), 'Preserve existing worker output/log'
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME='/usr/local/cuda',
                        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                        OMP_NUM_THREADS='4', MAX_JOBS='4', PYTHONUNBUFFERED='1', TORCH_CUDA_ARCH_LIST='12.0+PTX')
                    cache = Path('/data1/models/svdquant-wjq/research/cache/e051')
                    for key, name in {'HF_HOME':'hf','HF_HUB_CACHE':'hf/hub','HF_XET_CACHE':'hf/xet','XDG_CACHE_HOME':'xdg',
                        'TORCH_HOME':'torch','TRITON_CACHE_DIR':'triton','TORCHINDUCTOR_CACHE_DIR':'inductor',
                        'CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'torch_extensions','TMPDIR':'tmp'}.items():
                        dest = cache/name; dest.mkdir(parents=True, exist_ok=True); env[key] = str(dest)
                    env['PATH'] = str(Path(m['python']).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
                    end = min(deadline, time.time()+m['budget']['worker_timeout_seconds'])
                    command = [m['python'],'-u',str(RUNNER),'--manifest',str(args.manifest),'--prompt-ids',str(pid),
                               '--output',str(result),'--deadline-unix',str(end)]
                    stream = log.open('x'); streams.append(stream)
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    row = dict(prompt_id=pid, gpu=gpu, pid=proc.pid, command=command, status='running',
                        start_epoch=time.time(), deadline_epoch=end, gpu_immediately_before=last, log=str(log), output=str(result))
                    owned.append((proc,end,result,row)); report['workers'].append(row); pending.remove(group)
                    occupied.add(gpu); idle[gpu] = 0
                    save(output, report); print(f'Launched E051 prompt {pid} GPU{gpu} PID{proc.pid}', flush=True)
            save(output, report)
            time.sleep(2)
        # Collect workers that exited between the last poll and the loop condition.
        for proc, end, result, row in owned:
            rc = proc.wait(); row['returncode'] = rc
            assert rc == 0 and json.loads(result.read_text())['status'] == 'complete', f'Worker {proc.pid} failed'
            row.update(status='complete', result=record(result))
        assert len(owned) == 4
        report.update(status='complete', gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc()); raise
    finally:
        for proc, _, _, _ in owned:
            stop(proc)
        for stream in streams:
            stream.close()
        for proc, _, _, row in owned:
            row['returncode'] = proc.returncode
            if row['status'] == 'running':
                row['status'] = 'stopped_with_launcher'
        report['seconds'] = time.time()-started; save(output, report)


if __name__ == '__main__':
    main()
