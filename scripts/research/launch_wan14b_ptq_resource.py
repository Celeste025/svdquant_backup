#!/usr/bin/env python3
"""E052: wait for complete matched caches, then one bounded resource worker."""
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
MANIFEST = ROOT/'research_state/06_experiments/E052_wan14b_ptq_resource_manifest.json'
RUNNER = ROOT/'scripts/research/wan14b_ptq_resource.py'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    m = read(args.manifest); rd = Path(m['report_dir']); budget = m['budget']; gpu = budget['gpu']
    assert m['experiment'] == 'E052' and gpu == 4
    assert os.environ.get('TMUX'), 'Use named tmux'
    output, result, log = rd/'launcher.json', rd/'run.json', ROOT/'results/logs/E052_resource.log'
    assert not any(p.exists() for p in (output,result,log)), 'Preserve previous attempts'
    check_path = rd/'check.json'; check = read(check_path)
    assert check['status'] == 'complete' and check['cuda_initialized'] is False and check['model_weights_loaded'] is False
    sources = {r['file']: r for r in check['sources']}
    for source_path in (RUNNER, args.manifest.resolve()):
        actual = record(source_path)
        assert all(sources[str(source_path)][k] == actual[k] for k in ('file','bytes','sha256')), 'Source changed after CPU check'
    started = time.time(); wait_end = started + budget['prerequisite_wait_seconds']
    report = dict(experiment='E052', status='waiting_for_collection', start_epoch=started,
        supervisor_pid=os.getpid(), source=record(Path(__file__)), runner=record(RUNNER), manifest=record(args.manifest),
        cpu_check=record(check_path), prerequisite_deadline_epoch=wait_end, gpu=gpu,
        scope='Resource-only block0 full-budget smoothing/LR, no full PTQ/checkpoint/video/quality run')
    save(output, report); proc = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        collection_dir = Path(m['collection_dir'])
        paths = [collection_dir/n for n in ('collect_launcher.json','collection_summary.json','collection_summary_launcher.json')]
        while True:
            values = [read(p) if p.exists() else None for p in paths]
            for path,value in zip(paths,values):
                if value is not None:
                    assert value['status'] in ('running','complete'), f'Prior collection failed: {path}; do not restart'
            if all(v is not None and v['status']=='complete' for v in values):
                assert values[0]['manifest'] == m['collection_manifest'] and values[2]['returncode'] == 0
                assert len(values[0]['workers']) == 4 and all(w['status']=='complete' and w['returncode']==0 for w in values[0]['workers'])
                assert values[1]['cache_count'] == 64 and values[1]['prompt_count'] == 14
                report['collection'] = [record(p) for p in paths]
                report['prerequisite_seconds'] = time.time()-started; break
            if time.time() >= wait_end:
                raise TimeoutError('E052 prerequisite wait expired; original E050 is not stopped or restarted')
            time.sleep(10)
        ready = time.time(); launch_end = ready+budget['launcher_seconds']
        report.update(status='waiting_for_gpu', launch_start_epoch=ready, launcher_deadline_epoch=launch_end)
        save(output, report); previous = None
        while True:
            first = devices()[gpu]
            if first['memory_used_mib'] < 256 and first['utilization_pct'] == 0:
                time.sleep(2); second = devices()[gpu]
                if second['memory_used_mib'] < 256 and second['utilization_pct'] == 0:
                    last = devices()[gpu]
                    if last['memory_used_mib'] < 256 and last['utilization_pct'] == 0:
                        report['gpu_before'] = [first,second,last]; break
            if time.time()+budget['worker_seconds'] >= launch_end:
                raise TimeoutError('E052 GPU not free within launcher slack; no foreign process stopped')
            if first != previous:
                report['last_gpu_wait'] = dict(epoch=time.time(),state=first); save(output,report); previous=first
            time.sleep(5)
        end = min(launch_end,time.time()+budget['worker_seconds'])
        command = [m['python'],'-u',str(RUNNER),'--phase','run','--manifest',str(args.manifest),
                   '--output',str(result),'--deadline-unix',str(end)]
        env=environment(m['python'],gpu);env['DEEPCOMPRESSOR_WAN_GATED']='0'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(status='running',worker_pid=proc.pid,worker_deadline_epoch=end,command=command,log=str(log),output=str(result))
            save(output,report);print('E052 worker started',proc.pid,'GPU',gpu,flush=True)
            while proc.poll() is None:
                if time.time() >= end:
                    raise TimeoutError('E052 bounded worker resource deadline')
                time.sleep(2)
        report['returncode']=proc.returncode
        assert proc.returncode==0, 'E052 resource worker failed; preserve log and partial files'
        value=read(result);assert value['status']=='complete'
        report.update(status='complete',result=record(result))
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        if proc is not None:
            stop(proc);report['returncode']=proc.returncode
        report['seconds']=time.time()-started
        report['gpu_after']=devices().get(gpu)
        save(output,report);print(report['status'],report['seconds'],flush=True)


if __name__=='__main__':
    main()
