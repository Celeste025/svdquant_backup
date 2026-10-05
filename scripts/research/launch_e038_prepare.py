#!/usr/bin/env python3
"""E038 shared input preparation only, with a 600-second process-group limit."""
import json, os, signal, subprocess, time, traceback
from pathlib import Path
from resume_h3_plain_baseline import idle_stable
ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E038'
OUT=RD/'prepare_launcher.json'
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    m=json.loads((ROOT/'research_state/06_experiments/E038_center_video_manifest.json').read_text())
    check=json.loads((RD/'prepare_check.json').read_text())
    assert check['status']=='complete' and check['cuda_initialized'] is False
    assert not (RD/'prepare.json').exists()
    gpu=idle_stable(0)
    start=time.time(); deadline=start+m['budget']['prepare_wall_seconds']
    python=m['python_decode']; log=ROOT/'results/logs/E038_prepare.log'
    env=dict(os.environ,**m['environment'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env.update(TRITON_CACHE_DIR='/data1/models/svdquant-wjq/research/cache/triton',CUDA_CACHE_PATH='/data1/models/svdquant-wjq/research/cache/cuda')
    env['PATH']=str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
    command=[python,'-u',str(ROOT/'scripts/research/prepare_decode_h3_center_video.py'),'--phase','prepare','--deadline-unix',str(deadline)]
    report=dict(experiment='E038',phase='prepare',status='running',start_epoch=start,deadline_epoch=deadline,gpu_before=gpu,command=command,log=str(log))
    proc=None
    try:
        OUT.write_text(json.dumps(report,indent=2)+'\n')
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report['pid']=proc.pid; OUT.write_text(json.dumps(report,indent=2)+'\n')
            report['returncode']=proc.wait(timeout=max(1,deadline-time.time()))
        assert report['returncode']==0
        result=json.loads((RD/'prepare.json').read_text())
        assert result['status']=='complete' and result['actual_text_encoder_calls']==4 and len(result['cases'])==8
        report['status']='complete'
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc())
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        raise
    finally:
        report['seconds']=time.time()-start;OUT.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
if __name__=='__main__':main()
