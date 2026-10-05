#!/usr/bin/env python3
"""E045: two bounded same-state terminal interventions; no retries."""
import json
from pathlib import Path
import subprocess
import time
import traceback
from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_native_v2 import environment

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'research_state/06_experiments/E045_wan_terminal_intervention_manifest.json'
RUNNER=ROOT/'scripts/research/run_wan_terminal_intervention_v2.py'
RD=ROOT/'results/research/E045'
LOGS=ROOT/'results/logs'


def main():
    m=json.loads(MANIFEST.read_text()); output=RD/'launcher.json'
    assert not output.exists(), 'Keep prior launch artifacts'
    for group in m['worker_groups']:
        check=json.loads((RD/f"check_{group['case_id']}_v2.json").read_text())
        assert check['status']=='complete'
    started=time.time(); deadline=started+m['budget']['launcher_timeout_seconds']
    report=dict(experiment='E045',status='running',start_epoch=started,deadline_epoch=deadline,
                manifest=record(MANIFEST),runner=record(RUNNER),launcher=record(Path(__file__)),workers=[])
    owned=[]; streams=[]
    try:
        report['gpu_before']=devices();save(output,report)
        for group in m['worker_groups']:
            gpu=group['gpu']; case=group['case_id'];state=devices()[gpu]
            assert state['memory_used_mib']<256 and state['utilization_pct']==0, f'GPU {gpu} busy'
            result=RD/f'worker_{case}.json'; log=LOGS/f'E045_worker_{case}.log'
            assert not result.exists() and not log.exists(), 'Keep previous attempt'
            end=min(deadline,time.time()+m['budget']['worker_timeout_seconds'])
            cmd=[m['python'],'-u',str(RUNNER),'--case-id',case,'--manifest',str(MANIFEST),
                 '--output',str(result),'--deadline-unix',str(end),
                 '--resume-worker',str(RD/'attempt01'/f'worker_{case}.json')]
            stream=log.open('x');streams.append(stream)
            proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(m['python'],gpu),stdout=stream,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            item=dict(**group,pid=proc.pid,deadline_epoch=end,command=cmd,output=str(result),
                      log=str(log),status='running',gpu_at_start=state)
            owned.append((proc,item,result));report['workers'].append(item)
            save(output,report);print(f'Launched {case} GPU{gpu} PID{proc.pid}',flush=True)
        while any(proc.poll() is None for proc,_,_ in owned):
            for proc,item,result in owned:
                code=proc.poll()
                if code is None:
                    if time.time()>item['deadline_epoch']:raise TimeoutError(f"Worker {proc.pid} deadline")
                elif code!=0:raise RuntimeError(f'Worker {proc.pid} exited {code}; preserve log')
            if time.time()>deadline:raise TimeoutError('Launcher deadline')
            time.sleep(1)
        for proc,item,result in owned:
            assert proc.returncode==0, f'Worker {proc.pid} failed'
            r=json.loads(result.read_text());assert r['status']=='complete'
            item.update(status='complete',returncode=proc.returncode,result=record(result))
        report.update(status='complete',gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        for proc,item,_ in owned:
            stop(proc);item['returncode']=proc.returncode
            if item['status']=='running':item['status']='stopped_with_launcher'
        for stream in streams:stream.close()
        report['seconds']=time.time()-started;save(output,report)
        print(report['status'],report['seconds'],flush=True)


if __name__=='__main__':main()
