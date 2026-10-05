#!/usr/bin/env python3
"""E049: evaluate only eight new Wan14B BF16 videos, using the existing two evaluators."""
import json
from pathlib import Path
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_evaluation import base_env, PYTHON

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E049'
ADAPTER = ROOT/'scripts/research/wan14b_reference_evaluation.py'
MJ = ROOT/'scripts/research/eval_mjvideo_e021.py'
PLAN = ROOT/'research_state/06_experiments/E049_wan14b_reference_manifest.json'
ARM = 'wan14b_bf16'


def main():
    output = RD/'evaluation_launcher.json'
    assert not output.exists(), 'Preserve previous attempts'
    generation = json.loads((RD/'launcher.json').read_text())
    assert generation['status'] == 'complete' and generation['experiment'] == 'E049', 'Generation must complete first'
    check = json.loads((RD/'evaluation_check.json').read_text())
    assert check['status'] == 'complete' and not check['cuda_initialized']
    assert check['source'] == record(ADAPTER) and check['plan'] == record(PLAN), 'CPU structure binding changed'
    plan = json.loads(PLAN.read_text())
    logs = ROOT/'results/logs'; logs.mkdir(parents=True, exist_ok=True)
    start = time.time(); deadline = start+900
    report = dict(experiment='E049', status='running', start_epoch=start, deadline_epoch=deadline,
                  source=record(Path(__file__)), adapter=record(ADAPTER), mj_evaluator=record(MJ),
                  generation_launcher=record(RD/'launcher.json'), cpu_stages=[], workers=[],
                  new_videos=8, inherited_reference_evaluations=0)
    owned = []; streams = []
    try:
        env = base_env(); env['CUDA_VISIBLE_DEVICES'] = ''
        for stage, args in [('prepare', ['--phase', 'prepare']),
                            ('temporal_check', ['--phase', 'temporal', '--check-only'])]:
            log = logs/f'E049_{stage}.log'; assert not log.exists()
            cmd = [PYTHON, '-u', str(ADAPTER), *args]
            with log.open('x') as stream:
                began = time.time()
                done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
            report['cpu_stages'].append(dict(stage=stage, command=cmd, returncode=done.returncode,
                                              seconds=time.time()-began, log=str(log)))
            save(output, report); assert done.returncode == 0, f'{stage} failed; retain log'
        if time.time() >= deadline:
            raise TimeoutError('E049 evaluation deadline before GPU launch')
        before = devices(); report['gpu_before'] = before
        for gpu in (0, 1):
            assert before[gpu]['memory_used_mib'] < 256 and before[gpu]['utilization_pct'] == 0, f'GPU{gpu} busy'
        jobs = [(0, 'temporal_'+ARM, [PYTHON, '-u', str(ADAPTER), '--phase', 'temporal',
                  '--deadline-unix', str(deadline)], RD/f'temporal_{ARM}_scores.json'),
                (1, 'mjvideo', [PYTHON, '-u', str(MJ), '--manifest', str(RD/'evaluation_manifest.json'),
                  '--samples', plan['data_dir'], '--model', '/data1/models/svdquant-wjq/models/MJ-VIDEO-2B',
                  '--tokenizer', '/data1/models/svdquant-wjq/research/20261003/E021/tokenizer',
                  '--mjvideo-repo', '/data1/models/svdquant-wjq/third_party/MJ-Video',
                  '--output', str(RD/'mjvideo_scores.json'), '--variants', ARM,
                  '--video-template', '{variant}/{case_id}/video.mp4', '--num-segments', '8'], RD/'mjvideo_scores.json')]
        for gpu, metric, cmd, result in jobs:
            log = logs/f'E049_{metric}.log'; assert not log.exists() and not result.exists()
            env = base_env(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), MASTER_PORT=str(29749+gpu))
            stream = log.open('x'); streams.append(stream)
            began = time.time()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            owned.append((proc, metric, result))
            report['workers'].append(dict(metric=metric, gpu=gpu, pid=proc.pid, command=cmd,
                                          output=str(result), log=str(log), status='running', start_epoch=began))
            save(output, report); print(f'Launched {metric} GPU{gpu}, PID {proc.pid}', flush=True)
        while True:
            all_done = True
            for index, (proc, metric, result) in enumerate(owned):
                code = proc.poll()
                if code is None:
                    all_done = False
                elif report['workers'][index]['status'] == 'running':
                    item = report['workers'][index]; item['returncode'] = code
                    assert code == 0, f'{metric} failed; retain log'
                    value = json.loads(result.read_text())
                    if metric.startswith('temporal_'):
                        assert value['status'] == 'complete' and len(value['rows']) == 8
                    else:
                        assert len(value['results']) == 8 and all(set(r['variants']) == {ARM} for r in value['results'].values())
                    item.update(status='complete', result=record(result), seconds=time.time()-item['start_epoch'])
                    save(output, report)
            if all_done:
                break
            if time.time() > deadline:
                raise TimeoutError('E049 evaluation deadline')
            time.sleep(1)
        report.update(status='complete', gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        for proc, metric, result in owned:
            stop(proc)
        for stream in streams:
            stream.close()
        for item, (proc, metric, result) in zip(report['workers'], owned, strict=True):
            item['returncode'] = proc.returncode
            if item['status'] == 'running':
                item['status'] = 'stopped_with_launcher'
        report['seconds'] = time.time()-start
        save(output, report); print(report['status'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
