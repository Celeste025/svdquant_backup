#!/usr/bin/env python3
"""E043: bind every completed video, then supervise the two unchanged metrics."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from launch_vanilla_wan_reference import devices, stop, record, save

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E043'
PYTHON='/data1/models/svdquant-wjq/conda-envs/mjvideo/bin/python'
ADAPTER=ROOT/'scripts/research/vanilla_wan_evaluation.py'
MJ=ROOT/'scripts/research/eval_mjvideo_e021.py'


def base_env():
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='4',MAX_JOBS='4',PYTHONUNBUFFERED='1',MASTER_PORT='29643')
    cache=Path('/data1/models/svdquant-wjq/research/cache')
    for key,name in {'HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg','TORCH_HOME':'torch','TMPDIR':'tmp',
                     'TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'torch_extensions'}.items():
        p=cache/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    env['PATH']=str(Path(PYTHON).parent)+':/usr/local/cuda/bin:'+env.get('PATH','')
    return env


def main():
    output=RD/'evaluation_launcher.json'
    assert not output.exists(), 'Keep previous attempts'
    generation=json.loads((RD/'launcher.json').read_text())
    assert generation['status']=='complete', 'Full generation must finish first'
    started=time.time();deadline=started+900
    report=dict(experiment='E043',status='running',start_epoch=started,deadline_epoch=deadline,
                source=record(Path(__file__)),adapter=record(ADAPTER),mj_evaluator=record(MJ),
                generation_launcher=record(RD/'launcher.json'),workers=[],cpu_stages=[])
    owned=[];streams=[]
    try:
        env=base_env();env['CUDA_VISIBLE_DEVICES']=''
        for stage,args in [('prepare',['--phase','prepare']),('temporal_check',['--phase','temporal','--check-only'])]:
            log=ROOT/f'results/logs/E043_{stage}.log';assert not log.exists()
            cmd=[PYTHON,'-u',str(ADAPTER),*args]
            with log.open('x') as f:
                t=time.time();done=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
            report['cpu_stages'].append(dict(stage=stage,command=cmd,returncode=done.returncode,seconds=time.time()-t,log=str(log)))
            save(output,report);assert done.returncode==0, f'{stage} failed, preserve log'
        gpu_before=devices();report['gpu_before']=gpu_before
        for gpu in (0,1):
            assert gpu_before[gpu]['memory_used_mib']<256 and gpu_before[gpu]['utilization_pct']==0, f'GPU{gpu} busy'
        commands={
            'temporal':[PYTHON,'-u',str(ADAPTER),'--phase','temporal','--deadline-unix',str(deadline)],
            'mjvideo':[PYTHON,'-u',str(MJ),'--manifest',str(RD/'evaluation_manifest.json'),
                       '--samples','/data1/models/svdquant-wjq/research/20261003/E043',
                       '--model','/data1/models/svdquant-wjq/models/MJ-VIDEO-2B',
                       '--tokenizer','/data1/models/svdquant-wjq/research/20261003/E021/tokenizer',
                       '--mjvideo-repo','/data1/models/svdquant-wjq/third_party/MJ-Video',
                       '--output',str(RD/'mjvideo_scores.json'),'--variants','vanilla_wan_bf16',
                       '--video-template','{case_id}/video.mp4','--num-segments','8']}
        for gpu,(metric,cmd) in enumerate(commands.items()):
            log=ROOT/f'results/logs/E043_{metric}.log';assert not log.exists()
            result=RD/f'{metric}_scores.json';assert not result.exists()
            env=base_env();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            f=log.open('x');streams.append(f)
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            owned.append((p,metric,result));report['workers'].append(dict(metric=metric,gpu=gpu,pid=p.pid,command=cmd,log=str(log),output=str(result),status='running'))
            save(output,report);print(f'Launched {metric} GPU{gpu}, worker {p.pid}',flush=True)
        while True:
            complete=True
            for i,(p,metric,result) in enumerate(owned):
                code=p.poll()
                if code is None:complete=False
                elif report['workers'][i]['status']=='running':
                    report['workers'][i]['returncode']=code
                    assert code==0, f'{metric} failed, preserve log'
                    assert result.is_file(), f'{metric} missing result'
                    v=json.loads(result.read_text())
                    if metric=='temporal':assert v['status']=='complete' and len(v['rows'])==8
                    report['workers'][i].update(status='complete',result=record(result));save(output,report)
            if complete:break
            if time.time()>deadline:raise TimeoutError('E043 evaluation deadline')
            time.sleep(1)
        report.update(status='complete',gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        for p,metric,result in owned:stop(p)
        for f in streams:f.close()
        for item,(p,metric,result) in zip(report['workers'],owned,strict=True):
            item['returncode']=p.returncode
            if item['status']=='running':item['status']='stopped_with_launcher'
        report['seconds']=time.time()-started;save(output,report);print(report['status'],report['seconds'],flush=True)


if __name__=='__main__':main()
