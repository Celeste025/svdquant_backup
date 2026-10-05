#!/usr/bin/env python3
"""E044: supervise complete evaluation of the sixteen new native videos."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from launch_vanilla_wan_reference import devices, stop, record, save
from launch_vanilla_wan_evaluation import base_env, PYTHON

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E044'
ADAPTER = ROOT/'scripts/research/vanilla_wan_native_evaluation.py'
MJ = ROOT/'scripts/research/eval_mjvideo_e021.py'
ARMS = ('plain_nvfp4','svdquant_nvfp4')


def main():
    output=RD/'evaluation_launcher.json'
    assert not output.exists(), 'Preserve previous attempts'
    generation=json.loads((RD/'launcher.json').read_text())
    assert generation['status']=='complete', 'Generation must complete first'
    started=time.time();deadline=started+900
    report=dict(experiment='E044',status='running',start_epoch=started,deadline_epoch=deadline,
                source=record(Path(__file__)),adapter=record(ADAPTER),mj_evaluator=record(MJ),
                generation_launcher=record(RD/'launcher.json'),cpu_stages=[],workers=[])
    owned=[];streams=[]
    try:
        env=base_env();env['CUDA_VISIBLE_DEVICES']=''
        stages=[('prepare',['--phase','prepare'])]
        stages += [('check_'+arm,['--phase','temporal','--arm',arm,'--check-only']) for arm in ARMS]
        for stage,args in stages:
            log=ROOT/f'results/logs/E044_{stage}.log';assert not log.exists()
            cmd=[PYTHON,'-u',str(ADAPTER),*args]
            with log.open('x') as f:
                t=time.time();done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
            report['cpu_stages'].append(dict(stage=stage,command=cmd,returncode=done.returncode,seconds=time.time()-t,log=str(log)))
            save(output,report);assert done.returncode==0, f'{stage} failed; keep log'
        before=devices();report['gpu_before']=before
        for gpu in (0,1,5):
            assert before[gpu]['memory_used_mib']<256 and before[gpu]['utilization_pct']==0, f'GPU{gpu} busy'
        jobs=[]
        for gpu,arm in zip((0,1),ARMS):
            jobs.append((gpu,'temporal_'+arm,[PYTHON,'-u',str(ADAPTER),'--phase','temporal','--arm',arm,
                         '--deadline-unix',str(deadline)],RD/f'temporal_{arm}_scores.json'))
        jobs.append((5,'mjvideo',[PYTHON,'-u',str(MJ),'--manifest',str(RD/'evaluation_manifest.json'),
            '--samples','/data1/models/svdquant-wjq/research/20261003/E044',
            '--model','/data1/models/svdquant-wjq/models/MJ-VIDEO-2B',
            '--tokenizer','/data1/models/svdquant-wjq/research/20261003/E021/tokenizer',
            '--mjvideo-repo','/data1/models/svdquant-wjq/third_party/MJ-Video',
            '--output',str(RD/'mjvideo_scores.json'),'--variants',*ARMS,
            '--video-template','{variant}/{case_id}/video.mp4','--num-segments','8'],RD/'mjvideo_scores.json'))
        for gpu,metric,cmd,result in jobs:
            log=ROOT/f'results/logs/E044_{metric}.log';assert not log.exists() and not result.exists()
            env=base_env();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MASTER_PORT=str(29700+gpu))
            f=log.open('x');streams.append(f)
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            owned.append((p,metric,result));report['workers'].append(dict(metric=metric,gpu=gpu,pid=p.pid,
                command=cmd,output=str(result),log=str(log),status='running'))
            save(output,report);print(f'Launched {metric} GPU{gpu}, PID{p.pid}',flush=True)
        while True:
            all_done=True
            for i,(p,metric,result) in enumerate(owned):
                code=p.poll()
                if code is None:all_done=False
                elif report['workers'][i]['status']=='running':
                    report['workers'][i]['returncode']=code
                    assert code==0, f'{metric} failed; keep log'
                    value=json.loads(result.read_text())
                    if metric.startswith('temporal_'):assert value['status']=='complete' and len(value['rows'])==8
                    else:assert len(value['results'])==8 and all(set(r['variants'])==set(ARMS) for r in value['results'].values())
                    report['workers'][i].update(status='complete',result=record(result));save(output,report)
            if all_done:break
            if time.time()>deadline:raise TimeoutError('E044 evaluation deadline')
            time.sleep(1)
        report.update(status='complete',gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        for p,metric,result in owned:stop(p)
        for stream in streams:stream.close()
        for item,(p,metric,result) in zip(report['workers'],owned,strict=True):
            item['returncode']=p.returncode
            if item['status']=='running':item['status']='stopped_with_launcher'
        report['seconds']=time.time()-started;save(output,report);print(report['status'],report['seconds'],flush=True)


if __name__=='__main__':main()
