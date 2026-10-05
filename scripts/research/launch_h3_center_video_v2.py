#!/usr/bin/env python3
"""E038 attempt 2: use the CPU-gate-only generation fix; preserve attempt 1."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from launch_h3_plain_baseline import ROOT, sha, complete
from resume_h3_plain_baseline import idle_stable

MANIFEST = ROOT/'research_state/06_experiments/E038_center_video_manifest.json'
PREPARE = ROOT/'scripts/research/prepare_decode_h3_center_video.py'
GENERATE = ROOT/'scripts/research/run_h3_center_video_v2.py'


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def check(manifest_path, manifest):
    rd = Path(manifest['report_dir'])
    checks = [rd/'prepare_check.json', rd/'check_coarse16_v2.json']
    files = {str(p.resolve()): sha(p) for p in [Path(__file__), manifest_path,
        ROOT/'scripts/research/launch_h3_plain_baseline.py',
        ROOT/'scripts/research/resume_h3_plain_baseline.py']}
    for path in checks:
        result = complete(path)
        if result.get('cuda_initialized') is not False or result['manifest']['sha256'] != sha(manifest_path):
            raise RuntimeError(f'Invalid CPU check: {path}')
        files[str(path)] = sha(path)
        for row in result['sources'].values():
            if sha(row['file']) != row['sha256']:
                raise RuntimeError(f'Checked source changed: {row["file"]}')
            files[row['file']] = row['sha256']
    if str(PREPARE) not in files or str(GENERATE) not in files:
        raise RuntimeError('Both workers must have a current CPU check')
    return dict(experiment='E038', status='complete', phase='check', cuda_initialized=False,
        files=files, gpu_assignments=manifest['budget']['generation_gpus'],
        commands=dict(prepare=[manifest['python_decode'],str(PREPARE),'--phase','prepare'],
            denoise=[manifest['python'],str(GENERATE),'--phase','denoise','--arm','ARM'],
            decode=[manifest['python_decode'],str(PREPARE),'--phase','decode','--arm','ARM']),
        limits=manifest['budget'], scope='Generation only; evaluation has its own launcher. Complete prepare is reused.')


def validate_result(name, result_path, manifest):
    result = complete(result_path)
    if result['manifest']['sha256'] != sha(MANIFEST) or len(result['cases']) != 8:
        raise RuntimeError(f'Unexpected stage binding/allocation: {name}')
    if name == 'prepare':
        if result['actual_text_encoder_calls'] != 4 or result['actual_dit_calls'] != 0:
            raise RuntimeError('Preparation must call only four text encodings')
    elif name.startswith('denoise_'):
        if result['complete_dit_calls'] != 160 or sum(len(r['dit_calls']) for r in result['cases']) != 160:
            raise RuntimeError('Expected eight complete 20-call trajectories')
    elif result['actual_video_vae_calls'] != 8 or result['actual_audio_vae_calls'] != 8 or result['actual_dit_calls'] != 0:
        raise RuntimeError('Decoder allocation changed')
    if [r['case_id'] for r in result['cases']] != [r['case_id'] for r in manifest['cases']]:
        raise RuntimeError('Stage case order changed')
    return dict(result=str(result_path), result_sha256=sha(result_path))


def terminate(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run(args, manifest, report, output):
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run this long job inside a named tmux session')
    rd = Path(manifest['report_dir'])
    checked = complete(rd/'launcher_check_v2.json')
    for path, digest in checked['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Checked file changed before launch: {path}')
    prep_exists = (rd/'prepare.json').exists()
    seconds = 2400 if prep_exists else 3000
    started = time.time()
    deadline = min(started+seconds, args.deadline_unix or float('inf'))
    report.update(start_epoch=started, deadline_epoch=deadline, wall_budget_seconds=seconds,
        launcher_check_sha256=sha(rd/'launcher_check_v2.json'), stages=[], max_dit_calls=480)
    active = {}

    def spawn(phase, arm=None):
        name = phase if arm is None else phase+'_'+arm
        result_path = rd/(name+'.json')
        if result_path.exists():
            raise FileExistsError(result_path)
        gpu = manifest['budget']['generation_gpus'][arm or 'bf16']
        gpu_before = idle_stable(gpu)
        limit_key = 'prepare_wall_seconds' if phase == 'prepare' else phase+'_wall_seconds_per_arm'
        child_deadline = min(deadline, time.time()+manifest['budget'][limit_key])
        if child_deadline <= time.time():
            raise TimeoutError('E038 stage deadline already expired')
        python = manifest['python'] if phase == 'denoise' else manifest['python_decode']
        worker = GENERATE if phase == 'denoise' else PREPARE
        cmd = [python,'-u',str(worker),'--phase',phase,'--manifest',str(args.manifest),
            '--prepare-report',str(rd/'prepare.json'),'--deadline-unix',str(child_deadline)]
        if arm is not None:
            cmd += ['--arm',arm]
        env = dict(os.environ, **manifest['environment'], CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        env['PATH'] = str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
        env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
        env['CUDA_CACHE_PATH'] = '/data1/models/svdquant-wjq/research/cache/cuda'
        log = ROOT/'results/logs'/('E038_v2_'+name+'.log')
        entry = dict(name=name, phase=phase, arm=arm, status='running', command=cmd,
            gpu_before=gpu_before, log=str(log), start_epoch=time.time(), deadline_epoch=child_deadline)
        with log.open('x') as stream:
            proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        entry['pid'] = entry['process_group'] = proc.pid
        report['stages'].append(entry)
        active[name] = (proc, entry, result_path)
        save(report,output)
        print(f'Start {name} GPU{gpu}, pid {proc.pid}',flush=True)

    def wait_until_done(allow_decode):
        while active:
            for name, (proc, entry, result_path) in list(active.items()):
                code = proc.poll()
                if code is None:
                    if time.time() >= entry['deadline_epoch']:
                        raise TimeoutError(f'{name} exceeded its fixed deadline')
                    continue
                entry.update(returncode=code, end_epoch=time.time(), seconds=time.time()-entry['start_epoch'])
                del active[name]
                if code:
                    entry['status']='failed_stop'
                    raise RuntimeError(f'{name} exited {code}; see {entry["log"]}')
                entry.update(validate_result(name,result_path,manifest), status='complete')
                save(report,output)
                print(f'Complete {name}: {entry["seconds"]:.1f}s',flush=True)
                if allow_decode and entry['phase']=='denoise':
                    spawn('decode',entry['arm'])
            if active:
                time.sleep(.5)

    try:
        if prep_exists:
            report['stages'].append(dict(name='prepare', status='complete', reused=True,
                **validate_result('prepare',rd/'prepare.json',manifest)))
        else:
            spawn('prepare'); wait_until_done(False)
        for arm in manifest['arms']:
            spawn('denoise',arm)
        wait_until_done(True)
        report.update(status='complete', actual_dit_calls=480, videos=24)
    finally:
        for proc, entry, _ in active.values():
            terminate(proc)
            entry.update(status='terminated_after_failure',returncode=proc.returncode,end_epoch=time.time())
        report['seconds_total'] = time.time()-started
        save(report,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--deadline-unix',type=float)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    output=Path(manifest['report_dir'])/('launcher_check_v2.json' if args.phase=='check' else 'launcher_v2.json')
    if output.exists(): raise FileExistsError(output)
    report=dict(experiment='E038',phase=args.phase,status='running')
    def interrupted(signum,frame): raise RuntimeError(f'Launcher interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.phase=='check': report=check(args.manifest,manifest)
        else: run(args,manifest,report,output)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc()); raise
    finally:
        save(report,output)
        print(json.dumps(dict(status=report['status'],output=str(output))),flush=True)


if __name__=='__main__': main()
