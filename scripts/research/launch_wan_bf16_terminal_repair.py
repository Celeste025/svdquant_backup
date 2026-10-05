#!/usr/bin/env python3
"""E046: three queues, all eight same-state native/BF16-terminal comparisons."""
import json
from pathlib import Path
import subprocess
import time
import traceback
from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'research_state/06_experiments/E046_wan_terminal_repair_manifest.json'
RUNNER=ROOT/'scripts/research/run_wan_bf16_terminal_repair.py'
RD=ROOT/'results/research/E046'
LOGS=ROOT/'results/logs'


def main():
    m=json.loads(MANIFEST.read_text()); output=RD/'launcher.json'
    assert not output.exists(), 'Preserve previous launchers'
    runner=record(RUNNER)
    for group in m['worker_groups']:
        check=json.loads((RD/f"check_{group['prompt_id']}.json").read_text())
        assert check['status']=='complete'
        assert next(s for s in check['sources'] if s['file']==str(RUNNER))==runner, 'Checked runner changed'
    started=time.time();deadline=started+m['budget']['launcher_timeout_seconds']
    report=dict(experiment='E046',status='running',start_epoch=started,deadline_epoch=deadline,
                manifest=record(MANIFEST),runner=runner,launcher=record(Path(__file__)),
                workers=[],pending=list(m['worker_groups']))
    owned=[];streams=[];running={}
    try:
        report['gpu_before']=devices();save(output,report)
        while report['pending'] or running:
            for gpu,(proc,item,result) in list(running.items()):
                code=proc.poll()
                if code is None:
                    if time.time()>item['deadline_epoch']:raise TimeoutError(f'Worker {proc.pid} deadline')
                    continue
                item['returncode']=code
                assert code==0, f'Worker {proc.pid} failed {code}; preserve log'
                r=json.loads(result.read_text());assert r['status']=='complete' and len(r['cases'])==2
                item.update(status='complete',result=record(result));del running[gpu];save(output,report)
            if time.time()>deadline:raise TimeoutError('E046 launcher deadline')
            for group in list(report['pending']):
                gpu=group['gpu']
                if gpu in running:continue
                state=devices()[gpu]
                if state['memory_used_mib']>=256 or state['utilization_pct']!=0:continue
                pid=group['prompt_id'];result=RD/f'worker_{pid}.json';log=LOGS/f'E046_worker_{pid}.log'
                assert not result.exists() and not log.exists(), 'Preserve previous attempts'
                end=min(deadline,time.time()+m['budget']['worker_timeout_seconds'])
                cmd=[m['python'],'-u',str(RUNNER),'--prompt-id',str(pid),'--manifest',str(MANIFEST),
                     '--output',str(result),'--deadline-unix',str(end)]
                stream=log.open('x');streams.append(stream)
                proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(m['python'],gpu),stdout=stream,
                                      stderr=subprocess.STDOUT,start_new_session=True)
                item=dict(**group,pid=proc.pid,deadline_epoch=end,command=cmd,output=str(result),log=str(log),
                          status='running',gpu_at_start=state)
                owned.append((proc,item));running[gpu]=(proc,item,result)
                report['workers'].append(item);report['pending'].remove(group)
                save(output,report);print(f'Launched prompt {pid} GPU{gpu} PID{proc.pid}',flush=True)
            time.sleep(1)
        report.update(status='complete',gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        for proc,item in owned:
            stop(proc);item['returncode']=proc.returncode
            if item['status']=='running':item['status']='stopped_with_launcher'
        for stream in streams:stream.close()
        report['seconds']=time.time()-started;save(output,report)
        print(report['status'],report['seconds'],flush=True)


if __name__=='__main__':main()
