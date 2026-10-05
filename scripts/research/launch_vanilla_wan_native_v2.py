#!/usr/bin/env python3
"""E044: three GPU queues for the complete fixed plain/SVD reference suite."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from launch_vanilla_wan_reference import devices, stop, record, save

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E044_vanilla_wan_native_manifest.json'
RUNNER = ROOT/'scripts/research/run_vanilla_wan_native_v2.py'
RD = ROOT/'results/research/E044'
LOGS = ROOT/'results/logs'


def environment(python, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_HOME='/usr/local/cuda',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MAX_JOBS='4', PYTHONUNBUFFERED='1',
               SVDQUANT_DATA_ROOT='/data1/models/svdquant-wjq', TORCH_CUDA_ARCH_LIST='12.0+PTX')
    cache = Path('/data1/models/svdquant-wjq/research/cache')
    for key, sub in {'HF_HOME':'huggingface', 'XDG_CACHE_HOME':'xdg', 'TORCH_HOME':'torch',
                     'TORCHINDUCTOR_CACHE_DIR':'inductor', 'TRITON_CACHE_DIR':'triton',
                     'CUDA_CACHE_PATH':'cuda', 'TORCH_EXTENSIONS_DIR':'e044/torch_extensions',
                     'TMPDIR':'tmp'}.items():
        p = cache/sub; p.mkdir(parents=True, exist_ok=True); env[key] = str(p)
    env['FLASHINFER_WORKSPACE_BASE'] = str(cache/'flashinfer')
    env['PATH'] = str(Path(python).parent)+':/home/wjq/.conda/envs/convrot-wan/bin:/usr/local/cuda/bin:'+env.get('PATH','')
    return env


def main():
    m = json.loads(MANIFEST.read_text()); budget = m['budget']
    output = RD/'launcher.json'
    assert not output.exists(), 'Preserve previous launchers'
    assert RUNNER.is_file()
    for arm in m['arms']:
        check = json.loads((RD/f'check_{arm}_v2.json').read_text())
        assert check['status'] == 'complete', f'CPU check missing: {arm}'
    started = time.time(); deadline = started + budget['launcher_timeout_seconds']
    report = dict(experiment='E044',status='running',start_epoch=started,deadline_epoch=deadline,
                  manifest=record(MANIFEST),runner=record(RUNNER),launcher=record(Path(__file__)),workers=[],
                  pending=list(m['worker_groups']))
    owned = []; streams = []; running = {}
    try:
        report['gpu_before'] = devices()
        save(output,report)
        while report['pending'] or running:
            for gpu, (proc, item, result) in list(running.items()):
                code = proc.poll()
                if code is None:
                    if time.time() > item['deadline_epoch']:
                        raise TimeoutError(f'Worker {proc.pid} reached deadline')
                    continue
                item['returncode'] = code
                assert code == 0, f'Worker {proc.pid} exited {code}; keep log'
                r = json.loads(result.read_text())
                assert r['status'] == 'complete'
                assert len(r['cases']) == 2*len(item['prompt_ids'])
                item.update(status='complete',result=record(result))
                del running[gpu]; save(output,report)
            if time.time() > deadline:
                raise TimeoutError('E044 launcher deadline')
            for group in list(report['pending']):
                gpu = group['gpu']
                if gpu in running:
                    continue
                state = devices()[gpu]
                if state['memory_used_mib'] >= 256 or state['utilization_pct'] != 0:
                    # A just-finished CUDA context may need a moment to disappear.
                    # Also never launch over a newly arrived unrelated task.
                    continue
                arm = group['arm']; ids = group['prompt_ids']
                key = arm+'_'+'_'.join(map(str,ids))
                result = RD/f'worker_{key}.json'; log = LOGS/f'E044_worker_{key}.log'
                assert not result.exists() and not log.exists(), 'Keep prior worker artifacts'
                end = min(deadline,time.time()+budget['worker_timeout_seconds'])
                cmd = [m['python'],'-u',str(RUNNER),'--arm',arm,'--prompt-ids',*map(str,ids),
                       '--manifest',str(MANIFEST),'--output',str(result),'--deadline-unix',str(end)]
                env = environment(m['python'],gpu)
                env['RCM_RUNS_ROOT'] = str(Path(m['data_dir'])/'load_scratch'/key)
                stream = log.open('x'); streams.append(stream)
                proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item = dict(**group,pid=proc.pid,deadline_epoch=end,command=cmd,output=str(result),
                            log=str(log),status='running',gpu_at_start=state)
                report['workers'].append(item); report['pending'].remove(group)
                running[gpu] = (proc,item,result); owned.append((proc,item))
                save(output,report); print(f'Launched {key} GPU{gpu}, PID{proc.pid}',flush=True)
            time.sleep(1)
        report.update(status='complete',gpu_after=devices())
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc()); raise
    finally:
        for proc,item in owned:
            stop(proc); item['returncode']=proc.returncode
            if item['status']=='running': item['status']='stopped_with_launcher'
        for stream in streams: stream.close()
        report['seconds']=time.time()-started;save(output,report)
        print(report['status'],report['seconds'],flush=True)


if __name__ == '__main__': main()
