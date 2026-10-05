#!/usr/bin/env python3
"""E038 two independent evaluators, fixed 24 videos, one 1200s deadline."""
import argparse, json, os, signal, subprocess, time, traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E038'
DATA=Path('/data1/models/svdquant-wjq')
MANIFEST=ROOT/'research_state/06_experiments/E038_center_video_manifest.json'
OUT=RD/'evaluation_launcher.json'
def main():
    p=argparse.ArgumentParser();p.add_argument('--temporal-gpu',type=int,default=0);p.add_argument('--mj-gpu',type=int,default=1);a=p.parse_args()
    assert a.temporal_gpu!=a.mj_gpu
    if OUT.exists():raise FileExistsError(OUT)
    m=json.loads(MANIFEST.read_text())
    for arm in m['arms']:
        d=json.loads((RD/f'decode_{arm}.json').read_text())
        assert d['status']=='complete' and len(d['cases'])==8
    assert not (RD/'temporal_scores.json').exists() and not (RD/'mjvideo_scores.json').exists()
    idle={g:idle_stable(g) for g in (a.temporal_gpu,a.mj_gpu)}
    start=time.time();deadline=start+m['budget']['evaluation_wall_seconds'];python=m['python_evaluate']
    commands={
        'temporal':[python,'-u',str(ROOT/'scripts/research/evaluate_h3_center_video_temporal.py'),'--deadline-unix',str(deadline)],
        'mjvideo':[python,'-u',str(ROOT/'scripts/research/eval_mjvideo_e021.py'),'--manifest',str(MANIFEST),
            '--samples',m['data_dir'],'--model',str(DATA/'models/MJ-VIDEO-2B'),'--tokenizer',str(DATA/'research/20261003/E021/tokenizer'),
            '--mjvideo-repo',str(DATA/'third_party/MJ-Video'),'--output',str(RD/'mjvideo_scores.json'),
            '--variants',*m['arms'],'--num-segments','8','--video-template','decode_{variant}/{case_id}.mp4']}
    report=dict(experiment='E038',status='running',start_epoch=start,deadline_epoch=deadline,stages=[])
    active=[]
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
    try:
        for name,gpu in [('temporal',a.temporal_gpu),('mjvideo',a.mj_gpu)]:
            env=dict(os.environ,**m['environment'],CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='6',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT='29638')
            env.update(TRITON_CACHE_DIR=str(DATA/'research/cache/triton'),CUDA_CACHE_PATH=str(DATA/'research/cache/cuda'))
            env['PATH']=str(Path(python).parent)+':'+env.get('PATH','')
            log=ROOT/'results/logs'/f'E038_{name}.log';stream=log.open('x')
            proc=subprocess.Popen(commands[name],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry=dict(name=name,gpu=gpu,gpu_before=idle[gpu],pid=proc.pid,command=commands[name],log=str(log),status='running',start_epoch=time.time())
            active.append((proc,stream,entry));report['stages'].append(entry);save()
        while any(proc.poll() is None for proc,_,_ in active):
            if time.time()>=deadline:raise TimeoutError('E038 evaluation deadline')
            for proc,_,entry in active:
                rc=proc.poll()
                if rc is not None and entry['status']=='running':
                    entry.update(status='complete' if rc==0 else 'failed_preserved',returncode=rc,seconds=time.time()-entry['start_epoch']);save()
                    if rc:raise RuntimeError(f'{entry["name"]} exited {rc}')
            time.sleep(1)
        for proc,_,entry in active:
            assert proc.returncode==0
            if entry['status']=='running':entry.update(status='complete',returncode=0,seconds=time.time()-entry['start_epoch'])
        t=json.loads((RD/'temporal_scores.json').read_text());j=json.loads((RD/'mjvideo_scores.json').read_text())
        assert t['status']=='complete' and len(t['rows'])==24
        assert set(j['results'])=={r['case_id'] for r in m['cases']} and all(set(r['variants'])==set(m['arms']) for r in j['results'].values())
        report['status']='complete'
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        for proc,stream,entry in active:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                entry.update(status='terminated_by_launcher',returncode=proc.returncode)
            stream.close()
        report['seconds']=time.time()-start;save();print(json.dumps(report),flush=True)
if __name__=='__main__':main()
