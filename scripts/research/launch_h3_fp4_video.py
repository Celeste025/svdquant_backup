#!/usr/bin/env python3
"""E017 serial generation, decoding and eight-video scoring under one deadline."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from launch_h3_plain_baseline import ROOT, sha, complete
from resume_h3_plain_baseline import idle_stable

RD = ROOT/'results/research/E017'
MANIFEST = ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json'
OUTPUT = RD/'launcher.json'
RUNNER = ROOT/'scripts/research/run_h3_fp4_attention_video.py'
EVALUATOR = ROOT/'scripts/research/evaluate_h3_fp4_video.py'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    freeze = RD/'frozen_contract.json'
    frozen = complete(freeze)
    for path, digest in frozen['files'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen source/input changed: {path}')
    for path in [RD/'E017_check_block_mean.json', RD/'E017_visionreward_cpucheck.json']:
        checked = complete(path)
        if checked.get('cuda_initialized') is not False:
            raise RuntimeError(f'Expected a CPU-only check: {path}')
    manifest = json.loads(MANIFEST.read_text())
    start = time.time()
    deadline = start+manifest['budget']['wall_seconds']
    report = dict(experiment='E017', status='running', start_epoch=start,
        deadline_epoch=deadline, wall_budget_seconds=manifest['budget']['wall_seconds'],
        freeze_sha256=sha(freeze), max_dit_calls=80, expected_evaluation_answers=232, stages=[])
    stages = []
    for arm in manifest['arms']:
        stages.append(('denoise_'+arm, manifest['python'], RUNNER,
            ['--phase','denoise','--arm',arm], RD/f'E017_denoise_{arm}.json'))
    for arm in manifest['arms']:
        stages.append(('decode_'+arm, manifest['python_decode'], RUNNER,
            ['--phase','decode','--arm',arm], RD/f'E017_decode_{arm}.json'))
    stages.append(('visionreward', manifest['python_evaluate'], EVALUATOR,
        [], RD/'E017_visionreward.json'))
    try:
        for name, python, script, extra, result_path in stages:
            gpu = idle_stable(manifest['budget']['gpu'])
            remaining = deadline-time.time()
            if remaining <= 0:
                raise TimeoutError('Original E017 deadline expired')
            if result_path.exists():
                raise FileExistsError(result_path)
            command = [python, '-u', str(script), *extra, '--deadline-unix', str(deadline)]
            log = ROOT/'results/logs'/f'E017_{name}.log'
            env = dict(os.environ, **manifest['environment'],
                CUDA_VISIBLE_DEVICES=str(manifest['budget']['gpu']), PYTHONUNBUFFERED='1')
            env['PATH'] = str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
            env['TRITON_CACHE_DIR'] = '/data1/models/svdquant-wjq/research/cache/triton'
            env['CUDA_CACHE_PATH'] = '/data1/models/svdquant-wjq/research/cache/cuda'
            entry = dict(name=name,status='running',command=command,log=str(log),
                gpu_before=gpu,start_epoch=time.time())
            report['stages'].append(entry)
            OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
            print(f'Start {name} GPU{manifest["budget"]["gpu"]}; {log}',flush=True)
            with log.open('x') as stream:
                proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True)
                entry['pid'] = proc.pid
                OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
                try:
                    code = proc.wait(timeout=remaining)
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL)
                            proc.wait()
                    raise
            entry.update(returncode=code,end_epoch=time.time())
            entry['seconds'] = entry['end_epoch']-entry['start_epoch']
            if code:
                raise RuntimeError(f'{name} exited {code}')
            result = json.loads(result_path.read_text())
            if result['status'] not in ('complete','complete_with_invalid_answers'):
                raise RuntimeError(f'{name} is not complete: {result["status"]}')
            if name.startswith('denoise_'):
                actual = sum(len(case['dit_calls']) for case in result['cases'])
                if actual != 40 or len(result['cases']) != 2:
                    raise RuntimeError('Denoise call/case allocation changed')
                entry['dit_calls'] = actual
            elif name.startswith('decode_'):
                if len(result['cases']) != 2:
                    raise RuntimeError('Decode allocation changed')
                entry['videos'] = 2
            else:
                if len(result['results']) != 8 or any(len(case['answers']) != 29 for case in result['results']):
                    raise RuntimeError('Eight-video/all-29-question allocation changed')
                entry['evaluation_answers'] = 232
            entry.update(status='complete', report=str(result_path),report_sha256=sha(result_path))
            print(f'Complete {name} in {entry["seconds"]:.1f}s',flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        if report['stages'] and report['stages'][-1]['status']=='running':
            report['stages'][-1]['status']='failed_stop'
        report.update(status='failed_stop',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.time()-start
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
