#!/usr/bin/env python3
"""E077 fixed 2x2 composition, saved native exports, no calibration."""
import gc, json, os, sys, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import torch
import run_h3_native_paired_video as e010
import probe_h3_plain_baseline as base
import probe_h3_lowrank_initialization as refs
from h3_native_nvfp4 import NativeH3Linear, install_native_h3
ROOT=HERE.parents[1]
OUT=ROOT/'results/research/E077'
DATA=Path('/data1/models/svdquant-wjq/research/20261004/E077')
PLAN=ROOT/'research_state/06_experiments/E077_half_composition_plan.md'

def move(x):
    if isinstance(x,torch.Tensor): return x.cuda()
    if isinstance(x,dict): return {k:move(v) for k,v in x.items()}
    if isinstance(x,list): return [move(v) for v in x]
    if isinstance(x,tuple): return tuple(move(v) for v in x)
    return x

def save():
    (OUT/'run.json').write_text(json.dumps(report,indent=2)+'\n')

def guard():
    assert time.time()-start<600, 'time budget'
    assert report['calls']<=16
    assert torch.cuda.max_memory_allocated()<60*1024**3

@torch.inference_mode()
def run():
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    parent_path=ROOT/'results/research/E065b/evaluate.json'
    parent=json.loads(parent_path.read_text())
    assert parent['status']=='complete'
    report['parent']=base.file_record(parent_path)
    exports={a:json.loads(base.verify_file(parent['exports'][a]).read_text()) for a in ('restart','carry')}
    cases=[c for c in parent['cases'] if c['arm']=='restart']
    assert len(cases)==4
    sources={}; teachers={}
    for c in cases:
        k=c['case_id']+'/'+c['position']
        sources[k]=torch.load(base.verify_file(c['artifact']),map_location='cpu',weights_only=True,mmap=True)
        assert base.tree_signature(sources[k]['actual_dit_inputs'])==c['actual_dit_input_signature']
        teachers[k]=refs.load_reference(parent['references'][k]['bf16'],c['actual_dit_input_signature'])['raw_outputs']
    pipe=e010.load_h3_pipeline(full=False,vram_limit_gib=30.)
    pipe.load_models_to_device(['dit']); pipe.dit.eval()
    report['resident']=e010.make_h3_resident(pipe.dit)
    before=base.non_target_identity(pipe.dit)
    report['install']=install_native_h3(pipe.dit,Path(parent['exports']['restart']['file']).parent,
        activation_packer=base.pack_activation_fast,chunk_rows=1024)
    assert before==base.non_target_identity(pipe.dit)
    rows={a:{r['name']:r for r in m['layers']} for a,m in exports.items()}
    current={name:'restart' for name in rows['restart']}
    outputs={}
    for arm in ('RR','CR','CC','RC'):
        guard()
        replacements=[]
        for name in rows['restart']:
            which=arm[0 if int(name.split('.')[1])<25 else 1]
            desired='restart' if which=='R' else 'carry'
            if current[name]==desired: continue
            row=rows[desired][name]
            path=Path(parent['exports'][desired]['file']).parent/row['file']
            assert base.file_record(path)['sha256']==row['file_sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            old=pipe.dit.get_submodule(name)
            assert type(old) is NativeH3Linear
            assert torch.equal(old.smooth.cpu(),payload['tensors']['smooth'])
            new=NativeH3Linear.from_export(payload,device='cuda',activation_packer=base.pack_activation_fast,chunk_rows=1024)
            prefix,attr=name.rsplit('.',1)
            setattr(pipe.dit.get_submodule(prefix),attr,new)
            current[name]=desired
            replacements.append({'name':name,'source':desired,'sha256':row['file_sha256']})
            del old,new,payload
        assert before==base.non_target_identity(pipe.dit)
        report['replacements'][arm]=replacements
        gc.collect(); outputs[arm]={}
        for c in cases:
            guard(); k=c['case_id']+'/'+c['position']
            tree=move(sources[k]['actual_dit_inputs'])
            audit=base.RuntimeAudit(); audit.phase='native'
            report['attempted_calls']+=1; save()
            with audit.installed(),base.collect_fastpack_checks() as packs:
                raw=pipe.dit(*tree['args'],**tree['kwargs'])
                torch.cuda.synchronize()
            report['calls']+=1
            base.audit_contract(audit.row(),'svd')
            assert packs.summary['checked_calls']==200
            cpu,signatures=base.cpu_outputs(raw)
            exact=None
            if arm in ('RR','CC'):
                oldarm='restart' if arm=='RR' else 'carry'
                historical=next(r for r in parent['cases'] if r['arm']==oldarm and r['case_id']==c['case_id'] and r['position']==c['position'])
                ref=torch.load(base.verify_file(historical['artifact']),map_location='cpu',weights_only=True,mmap=True)
                exact=all(torch.equal(cpu[m],ref['raw_outputs'][m]) for m in cpu)
                assert exact,'endpoint replay failed'
            outputs[arm][k]=cpu
            path=DATA/(arm+'_'+k.replace('/','_')+'.pt')
            torch.save(cpu,path)
            metrics={m:float((v.double()-teachers[k][m].double()).square().sum()) for m,v in cpu.items()}
            report['cases'].append(dict(arm=arm,key=k,raw_sse=metrics,replay_exact=exact,artifact=base.file_record(path),runtime=audit.row()))
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(); save()
            print(arm,k,metrics,flush=True)
            del raw,tree
    report['comparisons']=[]
    for k in sources:
        row={'key':k}
        for m in ('video','audio'):
            vals={a:outputs[a][k][m].double() for a in outputs}
            sse={a:float((v-teachers[k][m].double()).square().sum()) for a,v in vals.items()}
            interaction=vals['CC']-vals['CR']-vals['RC']+vals['RR']
            row[m]=dict(sse=sse,mixed_vs_best={a:sse[a]/min(sse['RR'],sse['CC']) for a in ('CR','RC')},interaction_energy=float(interaction.square().sum()),endpoint_delta_energy=float((vals['CC']-vals['RR']).square().sum()))
        report['comparisons'].append(row)
    report['gate_pass']={a:all(r['video']['mixed_vs_best'][a]<=.95 for r in report['comparisons']) for a in ('CR','RC')}
    assert report['calls']==report['attempted_calls']==16
    report.update(status='complete',seconds=time.time()-start)
    save()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'run.json').exists()
    DATA.mkdir(parents=True,exist_ok=False)
    start=time.time()
    report=dict(status='running',calls=0,attempted_calls=0,cases=[],replacements={},driver=base.file_record(__file__),plan=base.file_record(PLAN))
    try: run()
    except BaseException:
        report.update(status='failed',error=traceback.format_exc(),seconds=time.time()-start);save();raise
