"""Independent CPU recomputation of the completed two-call LSE diagnosis."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,math,subprocess,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];RD=ROOT/'results/research/E019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    output=RD/'lse_diagnostic_independent.json';assert not output.exists()
    start=time.monotonic();torch.set_num_threads(6)
    run=json.loads((RD/'lse_diagnostic_v2_run.json').read_text());launch=json.loads((RD/'lse_diagnostic_v2_launcher.json').read_text())
    assert run['status']==launch['status']=='complete' and run['attention_calls']==2 and run['complete_dit_calls']==0
    assert [c['return_lse'] for c in run['calls']]==[False,True]
    assert not Path(f"/proc/{launch['pid']}").exists()
    for fname in ['frozen_contract.json','lse_diagnostic_frozen.json','lse_diagnostic_v2_frozen.json']:
        frozen=json.loads((RD/fname).read_text());assert all(sha(p)==d for p,d in frozen['files'].items())
    artifacts=[run['input_files']['reference'],*(c['output'] for c in run['calls']),run['calls'][1]['lse']]
    for a in artifacts:assert Path(a['file']).stat().st_size==a['bytes'] and sha(a['file'])==a['sha256']
    reference,a,b=[torch.load(x['file'],map_location='cpu',weights_only=True,mmap=True) for x in artifacts[:3]]
    assert a.shape==b.shape==reference.shape==(22539,56,128) and a.dtype==b.dtype==reference.dtype==torch.bfloat16
    assert torch.equal(a.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8))
    changed=0;err=0.;ref=0.;maximum=0.;details=[]
    for i in range(0,len(a),256):
        aa=a[i:i+256].float();bb=b[i:i+256].float();coords=torch.nonzero(aa!=bb)
        changed+=len(coords);d=bb.double()-aa.double();err+=float(d.square().sum());ref+=float(aa.double().square().sum());maximum=max(maximum,float(d.abs().max()))
        for c in coords.tolist():
            details.append(dict(index=[i+c[0],c[1],c[2]],false=float(aa[tuple(c)]),true=float(bb[tuple(c)])))
    lse=torch.load(artifacts[3]['file'],map_location='cpu',weights_only=True,mmap=True)
    assert bool(torch.isfinite(lse).all()) and lse.shape==(1,56,22656)
    reported=run['true_vs_false'];assert changed==reported['changed_elements'] and math.isclose(err,reported['error_energy'],rel_tol=1e-12) and maximum==reported['max_abs']
    assert not torch.cuda.is_initialized()
    result=dict(status='complete',scope='Full block0 output numeric recomputation; no partition result or quality claim',source_sha256=sha(__file__),
        input_reports={str(RD/n):sha(RD/n) for n in ['lse_diagnostic_v2_run.json','lse_diagnostic_v2_launcher.json']},artifacts=artifacts,
        e018_false_full_byte_exact=True,changed_elements=changed,elements=a.numel(),error_energy=err,reference_energy=ref,nmse=err/ref,rms=math.sqrt(err/a.numel()),max_abs=maximum,
        changed_coordinates=details,lse_finite=True,original_calls=1,diagnostic_calls=2,total_calls=3,complete_dit_calls=0,cuda_initialized=False,pid_exited=True,
        all_three_source_freezes_unchanged=True,seconds_total=time.monotonic()-start,
        gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','changed_elements','nmse','rms','max_abs','seconds_total']}))
if __name__=='__main__':main()
