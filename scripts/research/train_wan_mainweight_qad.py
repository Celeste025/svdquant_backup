#!/usr/bin/env python3
"""E020: ordinary main-weight QAD, same-packet native export, no online LR."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,random,time,traceback
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from diffusers import WanTransformer3DModel
ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E020'
MODEL=Path('/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer')
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E020')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
    return h.hexdigest()
def save_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def tree(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:tree(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(tree(v,device) for v in x)
    return x
def metric(x,y):
    x,y=x.float(),y.float();err=(x-y).double().square().sum().item();ref=y.double().square().sum().item()
    return dict(err2=err,ref2=ref,nmse=err/max(ref,1e-30),elements=y.numel())
def invoke(model,record):
    payload=record['payload']
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out=model(*tree(payload['input_args'],'cuda'),**tree(payload['input_kwargs'],'cuda'))
    return out[0] if isinstance(out,(tuple,list)) else out.sample

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,default=RD/'E020_cache_inventory.json')
    p.add_argument('--steps',type=int,default=64);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=20261003);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--data-dir',type=Path,default=DATA);p.add_argument('--output',type=Path,default=RD/'train_run.json')
    args=p.parse_args();assert not args.output.exists();args.data_dir.mkdir(parents=True,exist_ok=False)
    start=time.time();report=dict(experiment='E020',status='running',settings=vars(args).copy(),stages=[],updates=[],evaluations=[],complete_dit_calls=0,backward_steps=0)
    report['settings']={k:str(v) if isinstance(v,Path) else v for k,v in report['settings'].items()}
    def log():save_json(args.output,report)
    def budget():
        if time.time()>args.deadline_unix:raise TimeoutError('E020 original training deadline expired')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>60*1024**3:raise MemoryError('E020 60GiB peak allocation exceeded')
    def call(model,record):
        budget();report['complete_dit_calls']+=1;return invoke(model,record)
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5','Physical GPU5 required'
        assert torch.cuda.get_device_capability()==(12,0)
        import wan_mainweight_qad as qad
        from wan_nvfp4_fastpack import collect_fastpack_checks
        torch.set_num_threads(6);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
        inventory=json.loads(args.inventory.read_text());assert inventory['status']=='complete'
        files=[Path(__file__),Path(qad.__file__),ROOT/'scripts/research/wan_nvfp4_fastpack.py',ROOT/'scripts/research/wan_native_nvfp4.py',args.inventory,ROOT/'research_state/06_experiments/E020_wan_mainweight_qad_plan.md',MODEL/'config.json']
        report['sources']={str(f):sha(f) for f in files}
        wpath=MODEL/'diffusion_pytorch_model.safetensors';stat=wpath.stat()
        report['original_model']=dict(file=str(wpath),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        report['environment']=dict(python=os.sys.executable,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),sdpa='FLASH_ATTENTION',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        records={}
        for split in ('train','validation'):
            records[split]=[]
            for row in inventory[split]:
                assert Path(row['file']).stat().st_size==row['bytes'] and sha(row['file'])==row['sha256'],'Cache changed since inventory'
                payload=torch.load(row['file'],map_location='cpu',weights_only=False)
                assert str(payload['filename']).zfill(4)==str(row['prompt_id']).zfill(4) and int(payload['step'])==row['step']
                assert payload['input_args'][0].shape==(1,16,20,60,104)
                assert payload['input_kwargs']['encoder_hidden_states'].shape==(1,512,4096)
                records[split].append(dict(file=row['file'],prompt_id=row['prompt_id'],step=row['step'],payload=payload))
        assert len(records['train'])==16 and len(records['validation'])==8
        report['record_counts']={k:len(v) for k,v in records.items()};log()
        model=WanTransformer3DModel.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
        for par in model.parameters():par.requires_grad_(False)
        teacher_started=time.time();targets=[]
        with torch.no_grad():
            for split,rows in records.items():
                for record in rows:
                    out=call(model,record).detach().cpu();assert torch.isfinite(out).all()
                    record['target']=out;cached=record['payload'].get('outputs')
                    row=dict(split=split,file=record['file'],prompt_id=record['prompt_id'],step=record['step'])
                    if isinstance(cached,(tuple,list)) and torch.is_tensor(cached[0]):row['vs_cached_output']=metric(out,cached[0])
                    targets.append(dict(**row,target=out));print('teacher',split,record['prompt_id'],record['step'],flush=True)
        torch.save(targets,args.data_dir/'teacher_targets.pt');report['teacher_seconds']=time.time()-teacher_started;report['teacher_cache_comparison']=[{k:v for k,v in r.items() if k!='target'} for r in targets];del targets
        modules=qad.install_qad(model);params=[m.weight_master for m in modules.values()]
        assert len(params)==300 and sum(p.numel() for p in params)==1391984640
        assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in params}
        report['trainable_tensors']=len(params);report['trainable_parameters']=sum(p.numel() for p in params)
        model.enable_gradient_checkpointing();log()
        def evaluate(current,split,label,step,output_file=None):
            rows=[];outputs=[];t=time.time();current.eval()
            with torch.no_grad(),collect_fastpack_checks():
                for record in records[split]:
                    out=call(current,record).cpu();rows.append(dict(prompt_id=record['prompt_id'],step=record['step'],**metric(out,record['target'])))
                    if output_file:outputs.append(out)
            if output_file:torch.save(outputs,output_file)
            item=dict(label=label,optimizer_step=step,split=split,seconds=time.time()-t,rows=rows,
                      aggregate_nmse=sum(r['err2'] for r in rows)/sum(r['ref2'] for r in rows),mean_row_nmse=sum(r['nmse'] for r in rows)/len(rows))
            report['evaluations'].append(item);log();print('eval',label,step,split,item['aggregate_nmse'],flush=True)
        def export_native(step):
            t=time.time()
            with torch.no_grad(),collect_fastpack_checks():artifact=qad.export_packed(model)
            path=args.data_dir/f'packed_step{step:04d}.pt';torch.save(artifact,path)
            native=WanTransformer3DModel.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
            installed=qad.install_packed(native,artifact)
            del artifact;gc.collect();torch.cuda.empty_cache()
            before=sum(m.native_calls for m in installed.values())
            evaluate(native,'validation','native',step,args.data_dir/f'native_validation_step{step:04d}.pt')
            calls=sum(m.native_calls for m in installed.values())-before;assert calls==300*len(records['validation'])
            report.setdefault('exports',[]).append(dict(step=step,file=str(path),bytes=path.stat().st_size,sha256=sha(path),native_calls=calls,seconds=time.time()-t,module_count=len(installed),online_lowrank=False))
            del installed,native;gc.collect();torch.cuda.empty_cache();log()
        evaluate(model,'train','qdq',0)
        evaluate(model,'validation','qdq',0,args.data_dir/'qdq_validation_step0000.pt')
        export_native(0)
        optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.0,foreach=False)
        sampled_before=[p.detach().flatten()[::max(1,p.numel()//256)][:256].clone() for p in params]
        order=[]
        for epoch in range((args.steps+15)//16):
            ids=list(range(16));random.Random(args.seed+epoch).shuffle(ids);order.extend(ids)
        for step,idx in enumerate(order[:args.steps],1):
            budget();model.train();record=records['train'][idx];optimizer.zero_grad(set_to_none=True);t=time.time()
            with collect_fastpack_checks(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out=call(model,record).float();target=record['target'].to('cuda').float()
                loss=(out-target).square().sum()/target.square().sum().clamp_min(1e-20)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite QAD loss')
                loss.backward();report['backward_steps']+=1
                if step==1:
                    assert all(p.grad is not None for p in params),'Missing main-weight gradient'
                    norms=torch.stack([p.grad.detach().float().norm() for p in params])
                    assert torch.isfinite(norms).all() and (norms>0).all(),'Nonfinite/zero whole-matrix gradient'
                    report['first_gradient_norms']={name:float(v) for name,v in zip(modules,norms.cpu())}
                norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            row=dict(optimizer_step=step,record_index=idx,prompt_id=record['prompt_id'],denoise_step=record['step'],online_nmse=float(loss.detach()),gradient_norm=float(norm),seconds=time.time()-t,allocated_gib=torch.cuda.memory_allocated()/1024**3,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            report['updates'].append(row);del out,target,loss
            if step==1:
                changed=[int((p.detach().flatten()[::max(1,p.numel()//256)][:256]!=b).sum()) for p,b in zip(params,sampled_before)]
                assert sum(changed)>0,'Optimizer changed no sampled master value'
                report['first_update_sampled_changed_counts']=dict(zip(modules,changed));del sampled_before
            log();print('train',json.dumps(row),flush=True)
            if step in {16,32,args.steps}:
                evaluate(model,'validation','qdq',step,args.data_dir/f'qdq_validation_step{step:04d}.pt')
                checkpoint=dict(step=step,model={k:v.detach().cpu() for k,v in model.state_dict().items()},settings=report['settings'],inventory_sha256=sha(args.inventory),sources=report['sources'])
                checkpoint_path=args.data_dir/f'master_step{step:04d}.pt';torch.save(checkpoint,checkpoint_path);del checkpoint
                report.setdefault('checkpoints',[]).append(dict(step=step,file=str(checkpoint_path),bytes=checkpoint_path.stat().st_size));log()
        optimizer.zero_grad(set_to_none=True)
        optpath=args.data_dir/f'optimizer_step{args.steps:04d}.pt';torch.save(optimizer.state_dict(),optpath);report['optimizer_checkpoint']=dict(file=str(optpath),bytes=optpath.stat().st_size)
        del optimizer;gc.collect();torch.cuda.empty_cache()
        evaluate(model,'train','qdq',args.steps)
        export_native(args.steps)
        report['status']='complete'
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds_total']=time.time()-start
        if torch.cuda.is_initialized():report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        log();print('E020',report['status'],report['seconds_total'],flush=True)
if __name__=='__main__':main()
