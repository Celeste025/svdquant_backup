#!/usr/bin/env python3
"""Two-call diagnostic supervisor; original E019 artifacts are immutable."""
import json, os, signal, subprocess, time, traceback
from pathlib import Path
from launch_h3_plain_baseline import ROOT, sha, complete
from resume_h3_plain_baseline import idle_stable
RD=ROOT/'results/research/E019'
def main():
    output=RD/'lse_diagnostic_v2_launcher.json'
    if output.exists(): raise FileExistsError(output)
    frozen=complete(RD/'lse_diagnostic_v2_frozen.json')
    for p,d in frozen['files'].items():
        if sha(p)!=d: raise RuntimeError(f'Changed source {p}')
    prior=json.loads((ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    python=prior['python']; started=time.time()
    report=dict(status='running',start_epoch=started,original_failed_calls=1,max_diagnostic_calls=2,
                max_total_calls=3,freeze_sha256=sha(RD/'lse_diagnostic_v2_frozen.json'))
    try:
        report['gpu_before']=idle_stable(5)
        deadline=time.time()+300
        env=dict(os.environ,**prior['environment'],CUDA_VISIBLE_DEVICES='5',PYTHONUNBUFFERED='1')
        env['PATH']=str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
        env['TRITON_CACHE_DIR']='/data1/models/svdquant-wjq/research/cache/triton'
        env['CUDA_CACHE_PATH']='/data1/models/svdquant-wjq/research/cache/cuda'
        command=[python,'-u',str(ROOT/'scripts/research/E019_lse_diagnostic_v2.py'),'--phase','run','--deadline-unix',str(deadline)]
        report.update(command=command,deadline_epoch=deadline)
        with (ROOT/'results/logs/E019_lse_diagnostic_v2.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['pid']=proc.pid;output.write_text(json.dumps(report,indent=2)+'\n')
            try: code=proc.wait(timeout=max(.01,deadline-time.time()))
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise
        report['returncode']=code
        if code: raise RuntimeError(f'Diagnostic exited {code}')
        result=complete(RD/'lse_diagnostic_v2_run.json')
        if result['attention_calls']!=2: raise RuntimeError('Wrong diagnostic call count')
        report.update(status='complete',result_sha256=sha(RD/'lse_diagnostic_v2_run.json'))
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds_total']=time.time()-started;output.write_text(json.dumps(report,indent=2)+'\n')
        print(report['status'],report['seconds_total'],flush=True)
if __name__=='__main__':main()
