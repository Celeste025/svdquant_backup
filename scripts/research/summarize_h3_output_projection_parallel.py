#!/usr/bin/env python3
"""E039 independent CPU reducer: four output-return / projection contracts.

Only final projected outputs and small state artifacts are loaded. Native
rounding differences are measured; equality to BF16-return is not a gate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E039'
ARMS=('bf16_return','fp8_return','row_parallel','fp4_side')
REAL=(11264,11328);VALID=(11264,11275);PHYSICAL=(11264,11392)
HALF,DOUT,R,SLOT=3584,5376,32,11392
COUNTERS=('boundary','native_gemm','lr_down','lr_up','smooth','pack','fp8_quant','fp8_dequant',
          'amax_reduce','all_to_all','all_reduce','reduce_scatter')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def record(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return dict(file=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def dist(values):
    return dict(values=values,median=statistics.median(values),min=min(values),max=max(values))


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def load(ref):
    require(record(ref['file'])==ref,'Final artifact file binding')
    return torch.load(ref['file'],map_location='cpu',weights_only=True,mmap=True)


def tensor_check(t,ref,shape):
    require(list(t.shape)==list(shape) and t.dtype==torch.bfloat16 and bool(t.isfinite().all()),'Output/state tensor contract')
    require(ref['shape']==list(shape) and ref['dtype']==str(t.dtype),'Tensor record shape/dtype')
    if 'sha256' in ref:require(tensor_sha(t)==ref['sha256'],'Saved tensor provenance')
    if 'finite' in ref:require(ref['finite'],'Saved finite receipt')


def metrics(pairs):
    sums=dict(error2=0.,reference2=0.,value2=0.,dot=0.,elements=0,max_abs=0.)
    for value,reference in pairs:
        require(value.shape==reference.shape,'Comparison shapes')
        for start in range(0,len(value),256):
            a=value[start:start+256].double();b=reference[start:start+256].double();e=a-b
            sums['error2']+=float(e.square().sum());sums['reference2']+=float(b.square().sum())
            sums['value2']+=float(a.square().sum());sums['dot']+=float((a*b).sum())
            sums['elements']+=e.numel();sums['max_abs']=max(sums['max_abs'],float(e.abs().max()))
    n=sums['elements'];den=sums['reference2'];cosden=math.sqrt(den*sums['value2'])
    return dict(**sums,nmse=sums['error2']/den if den else None,
        error_rms=math.sqrt(sums['error2']/n),error_l2=math.sqrt(sums['error2']),
        reference_rms=math.sqrt(den/n),cosine=sums['dot']/cosden if cosden else None)


def expected_counts(arm):
    values=dict.fromkeys(COUNTERS,0)
    distributed=arm in ('row_parallel','fp4_side')
    values.update(boundary=1,native_gemm=2 if arm=='row_parallel' else 1,
        lr_down=2 if distributed else 1,lr_up=1,smooth=1,pack=2 if distributed else 1,
        fp8_quant=2 if arm=='fp8_return' else 0,fp8_dequant=2 if arm=='fp8_return' else 0,
        amax_reduce=2 if distributed else 0,all_to_all=0 if arm=='row_parallel' else 1,
        all_reduce=int(distributed),reduce_scatter=int(arm=='row_parallel'))
    return values


def field_specs(arm,destination):
    m=REAL[destination]
    if arm=='bf16_return':return [('o',[m,HALF],'torch.bfloat16',m*HALF*2)]
    if arm=='fp8_return':return [('codes',[m,HALF],'torch.uint8',m*HALF),('scale',[1],'torch.float32',4)]
    return [('codes',[m,HALF//2],'torch.uint8',m*HALF//2),
        ('sf',[PHYSICAL[destination]*(HALF//16)],'torch.float8_e4m3fn',PHYSICAL[destination]*(HALF//16)),
        ('down',[m,R],'torch.bfloat16',m*R*2)]


def collective_ledger(pair,arm):
    rows=[]
    for rank,receipt in enumerate(pair):
        cs=receipt['collectives'];distributed=arm in ('row_parallel','fp4_side')
        require(len(cs)==1+int(distributed),'Collective count')
        row=dict(a2a_remote_send_bytes=0,a2a_remote_receive_bytes=0,a2a_self_bytes=0,
                 allreduce_logical_peer_send_bytes=0,reduce_scatter_logical_peer_send_bytes=0)
        if distributed:
            c=cs[0]
            require(c['kind']=='all_reduce_max' and c['dtype']=='torch.float32' and c['shape']==[2],'Global domain reduction')
            require(c['logical_peer_send_bytes']==c['logical_peer_receive_bytes']==8,'Two destination amax scalars')
            require(c['real_rows']==list(REAL) and c['includes_model_padding_rows']==53,'Real padding included in global')
            row['allreduce_logical_peer_send_bytes']=8
        c=cs[-1]
        if arm=='row_parallel':
            size=SLOT*(DOUT+R)*2
            require(c['kind']=='reduce_scatter_sum' and c['dtype']=='torch.bfloat16','Natural BF16 RS')
            require(c['input_shape']==[2,SLOT,DOUT+R] and c['output_shape']==[SLOT,DOUT+R],'RS shape')
            require(c['extra_slot_zero_rows']==[128,64] and c['main_features']==DOUT and c['down_features']==R,'RS real/transport rows')
            require(c['logical_peer_send_bytes']==c['logical_peer_receive_bytes']==size,'RS logical payload')
            row['reduce_scatter_logical_peer_send_bytes']=size
        else:
            peer=pair[1-rank]['collectives'][-1]
            expected=[sum(f[3] for f in field_specs(arm,d)) for d in range(2)]
            receive=[expected[rank]]*2
            require(c['kind']=='all_to_all' and c['dtype']=='torch.uint8' and c['label']==arm,'A2A format')
            require(c['send_splits']==expected and c['receive_splits']==receive,'A2A bytes per destination')
            require(c['input_numel']==sum(expected) and c['output_numel']==sum(receive),'Wire byte lengths')
            require(c['remote_send_bytes']==expected[1-rank]==peer['receive_splits'][rank]
                and c['remote_receive_bytes']==receive[1-rank]==peer['send_splits'][rank],'Peer symmetry')
            require(c['self_send_bytes']==expected[rank],'Self payload')
            for destination,fields in enumerate(c['fields']):
                require(fields==[dict(name=n,shape=s,dtype=t,bytes=b) for n,s,t,b in field_specs(arm,destination)],'Actual wire fields')
            row.update(a2a_remote_send_bytes=c['remote_send_bytes'],a2a_remote_receive_bytes=c['remote_receive_bytes'],
                       a2a_self_bytes=c['self_send_bytes'])
        rows.append(row)
    combined={k:sum(r[k] for r in rows) for k in rows[0]}
    require(combined['a2a_remote_send_bytes']==combined['a2a_remote_receive_bytes'],'Two-rank application symmetry')
    return dict(per_rank=rows,combined=combined,
        byte_scope='A2A actual application wire splits; MAX/RS logical peer payload separately, not NCCL link traffic. Self and receive are not added to remote-send totals.')


def reduce(args):
    rank_paths=[args.report_dir/f'run_rank{rank}.json' for rank in range(2)]
    ranks=[json.loads(p.read_text()) for p in rank_paths]
    expected={k:13*sum(expected_counts(a)[k] for a in ARMS) for k in COUNTERS}
    for rank,r in enumerate(ranks):
        require(r['status']=='complete' and r['rank']==rank and r['world_size']==2,'Wait for both complete ranks')
        require(r['arms']==list(ARMS) and r['actual_dit_calls']==0 and r['control_barrier_calls']==54,'Allocation/barriers')
        require(r['actual_counts']==expected,'Whole-run actual counters')
    for key in ('geometry','sources','input','weight','numerical_contracts'):
        require(ranks[0][key]==ranks[1][key],'Two-rank shared contract: '+key)
    geo=ranks[0]['geometry']
    require(geo['real_total']==22592 and geo['main_valid']==22539 and geo['model_padding_rows']==53
            and geo['real_token_lengths']==list(REAL) and geo['valid_token_lengths']==list(VALID),'True model support')
    prep_path=args.report_dir/'prepare.json';prep=json.loads(prep_path.read_text())
    require(prep['status']=='complete' and prep['actual_sdpa_calls']==2 and prep['actual_dit_calls']==0,'Fresh actual attention output preparation')
    require(prep['artifact']==ranks[0]['input'] and prep['model_total']==22592 and prep['model_padding']==53,'Prepared real O provenance')
    require(record(ranks[0]['sources']['runner']['file'])==ranks[0]['sources']['runner'],'Executed runner source')
    result=dict(experiment='E039',status='running',cpu_only=True,rank_reports=[record(p) for p in rank_paths],
        prepare=record(prep_path),sources=ranks[0]['sources'],weight_reference=ranks[0]['weight'],
        input_reference=ranks[0]['input'],geometry=geo,numerical_contracts=ranks[0]['numerical_contracts'],arms={},
        actual_boundaries=104,actual_native_gemms=130,actual_dit_calls=0,prepare_sdpa_calls=2,
        rank_receipts=[{k:r[k] for k in ('rank','actual_counts','environment','resident_baseline','after_warmup_baseline','released_allocated_bytes')} for r in ranks],
        limitations=['BF16-return is the original native SVD projection reference, not an all-BF16 model or quality oracle.',
            'Real model padding53 is reported separately; artificial SF/RS zero slots are counted as transport cost, excluded from output accuracy.',
            'LR/main partial arithmetic and FP8 alter numerical contracts. No bitwise scientific gate or quality inference.',
            'Shared resident full and sliced weights/inputs are benchmark fixtures. Peak and increment are both reported; not full-model deployment memory.',
            'Synchronized wall per round uses max across ranks. CUDA durations are retained per rank and never added to wall or across stages.',
            'One actual layer/state and two GPUs; not a video-quality, novelty, or scaling result.'])
    outputs={};states={}
    for arm in ARMS:
        entries=[next(e for e in r['benchmarks'] if e['arm']==arm) for r in ranks];ledger=None
        for label,count in [('warmups',3),('repeats',10)]:
            require(all(len(e[label])==count for e in entries),'Warmup/repeat count')
            for i in range(count):
                pair=[e[label][i] for e in entries];shift=i%4;order=ARMS if label=='warmups' else ARMS[shift:]+ARMS[:shift]
                for row in pair:
                    require(row['round']==i and row['order']==list(order) and row['position']==order.index(arm),'Paired execution order')
                    require(row['actual_counts']==expected_counts(arm),'Per-boundary actual counters')
                    require(all(math.isfinite(row[k]) and row[k]>0 for k in ('wall_ms','cuda_ms')),'Finite durations')
                    require(row['pack_flag_checks']['calls']==expected_counts(arm)['pack'] and row['pack_flag_checks']['invalid_calls']==0,'Pack flags')
                    for k in ('allocated','reserved'):
                        require(row['incremental_peak_'+k]==row['after']['peak_'+k]-row['before'][k],'Memory increment definition')
                current=collective_ledger(pair,arm)
                if ledger is None:ledger=current
                else:require(ledger==current,'Fixed wire ledger repeated')
        walls=[[r['wall_ms'] for r in e['repeats']] for e in entries]
        events=[[r['cuda_ms'] for r in e['repeats']] for e in entries]
        memory=[];parts=[];small=[]
        for rank,e in enumerate(entries):
            require(e['status']=='complete' and math.isclose(e['median_wall_ms'],statistics.median(walls[rank]),rel_tol=1e-12),'Per-rank timing median')
            mem={}
            for k in ('allocated','reserved'):
                mem['resident_before_'+k]=dist([r['before'][k] for r in e['repeats']])
                mem['peak_'+k]=dist([r['after']['peak_'+k] for r in e['repeats']])
                mem['incremental_peak_'+k]=dist([r['incremental_peak_'+k] for r in e['repeats']])
            memory.append(mem)
            output=load(e['output']['artifact']);tensor_check(output,e['output']['tensor'],(REAL[rank],DOUT));parts.append(output)
            state=load(e['state']['artifact']);tensor_check(state['down'],e['state']['tensors']['down'],(REAL[rank],R))
            g=state['activation_globals'];require(g.dtype==torch.float32 and bool(g.isfinite().all()) and bool((g>0).all()),'Positive activation globals')
            require(tensor_sha(g)==e['state']['tensors']['activation_globals']['sha256'],'Global state tensor provenance')
            small.append(state)
        outputs[arm]=parts;states[arm]=small
        result['arms'][arm]=dict(synchronized_wall_pair_max_ms=dist([max(a,b) for a,b in zip(*walls)]),
            per_rank_wall_ms=[dist(w) for w in walls],per_rank_cuda_ms=[dist(v) for v in events],memory_per_rank=memory,
            max_single_device_peak_allocated=max(m['peak_allocated']['max'] for m in memory),
            max_single_device_peak_reserved=max(m['peak_reserved']['max'] for m in memory),communication_per_boundary=ledger,
            output_parts=[e['output'] for e in entries],state_parts=[e['state'] for e in entries],
            activation_globals=[s['activation_globals'].tolist() for s in small])
    for arm in ARMS:
        pairs=list(zip(outputs[arm],outputs['bf16_return']))
        result['arms'][arm]['error_vs_bf16_return']=dict(
            valid=metrics([(v[:VALID[i]],b[:VALID[i]]) for i,(v,b) in enumerate(pairs)]),
            real_model_padding=metrics([(pairs[1][0][VALID[1]:],pairs[1][1][VALID[1]:])]),
            all_real=metrics(pairs))
        result['arms'][arm]['down_error_vs_bf16_return']=dict(
            valid=metrics([(states[arm][i]['down'][:VALID[i]],states['bf16_return'][i]['down'][:VALID[i]]) for i in range(2)]),
            real_model_padding=metrics([(states[arm][1]['down'][VALID[1]:],states['bf16_return'][1]['down'][VALID[1]:])]))
    result['comparisons']={}
    for reference in ('bf16_return','fp8_return','row_parallel'):
        a=result['arms']['fp4_side'];b=result['arms'][reference]
        va=a['synchronized_wall_pair_max_ms']['values'];vb=b['synchronized_wall_pair_max_ms']['values']
        result['comparisons']['fp4_side_vs_'+reference]=dict(
            median_wall_ratio=statistics.median(va)/statistics.median(vb),paired_round_wall_ratios=[x/y for x,y in zip(va,vb)],
            valid_nmse_delta=a['error_vs_bf16_return']['valid']['nmse']-b['error_vs_bf16_return']['valid']['nmse'],
            pad_nmse_delta=a['error_vs_bf16_return']['real_model_padding']['nmse']-b['error_vs_bf16_return']['real_model_padding']['nmse'],
            max_peak_allocated_ratio=a['max_single_device_peak_allocated']/b['max_single_device_peak_allocated'])
    result.update(status='complete',cuda_initialized=torch.cuda.is_initialized(),reducer_source=record(__file__))
    require(not result['cuda_initialized'],'CPU-only reduction')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=RD);p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or args.report_dir/'independent_summary.json'
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA externally');require(not args.output.exists(),'Preserve old summary/failure')
    global torch
    import torch
    torch.set_num_threads(6);start=time.monotonic();result=dict(experiment='E039',status='failed_stop',reducer_source=record(__file__))
    try:result=reduce(args)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:
        result['seconds']=time.monotonic()-start;save(args.output,result)
        print(json.dumps(dict(status=result['status'],output=str(args.output),sha256=record(args.output)['sha256'])))


if __name__=='__main__':main()
