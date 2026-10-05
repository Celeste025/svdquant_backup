#!/usr/bin/env python3
"""E037: unchanged distributed coarse16 versus global-Q/T1 strong baseline.

Reuses E036 transport and coarse execution. No new attention/packing kernel.
"""
import argparse
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import bench_h3_sequence_parallel_correction as base

common=base.common
ROOT=base.ROOT
RD=ROOT/'results/research/E037'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E037')
PLAN=ROOT/'research_state/06_experiments/E037_sequence_parallel_global_plan.md'
N,NP,H,D=base.N,base.NP,base.H,base.D
LENGTHS,VALID,STARTS=base.LENGTHS,base.VALID,base.STARTS
ARMS=('fp4_table16','fp4_global')


def global_wire_specs(length,heads):
    """Same packets as E036, with one correction row instead of sixteen."""
    rows=base.wire_specs('fp4_table16',length,heads)
    name,shape,dtype=rows[-1];assert name=='table'
    return rows[:-1]+[(name,(heads,1,length),dtype)]


def local_qk_sums(q,k,rank):
    import torch
    return torch.stack((q.sum(1,keepdim=True,dtype=torch.float32),
        k[:,:VALID[rank]].sum(1,keepdim=True,dtype=torch.float32)),dim=0)


def means_from_sums(sums):
    import torch
    return (sums[0]/NP).to(torch.bfloat16),(sums[1]/N).to(torch.bfloat16)


class Execution(base.Execution):
    def __init__(self,rank,module,ids,budget,official):
        super().__init__(rank,module,ids,budget)
        self.official=official;self.counts['q_sum_ops']=0

    def global_means(self,q,k):
        import torch.distributed as dist
        self.counts['q_sum_ops']+=1;self.counts['k_sum_ops']+=1
        sums=local_qk_sums(q,k,self.rank)
        self.counts['all_reduce']+=1;size=sums.numel()*sums.element_size()
        self.collectives.append(dict(kind='all_reduce_sum',label='padded_Q_and_valid_K_sums',dtype=str(sums.dtype),
            shape=list(sums.shape),input_numel=sums.numel(),output_numel=sums.numel(),element_size=sums.element_size(),
            logical_peer_send_bytes=size,logical_peer_receive_bytes=size,self_send_bytes=0,
            fields=dict(padded_Q_sum=dict(bytes=size//2,denominator=NP),valid_K_sum=dict(bytes=size//2,denominator=N)),
            byte_scope='Actual combined FP32 reduction tensor; world2 one-peer logical payload, not measured NCCL protocol traffic.'))
        self.budget();dist.all_reduce(sums,op=dist.ReduceOp.SUM)
        return means_from_sums(sums)

    def global_forward_exchange(self,fields):
        wire,send_sizes=base.encode_head_payload(fields,H)
        sizes=[base.spec_bytes(global_wire_specs(length,H//2)) for length in LENGTHS]
        desc={name:dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
              for name,value in fields.items()}
        received=self.a2a(wire,send_sizes,sizes,'forward_fp4_global',desc)
        parts=[];offset=0
        for size,length in zip(sizes,LENGTHS):
            parts.append(base.decode_payload(received[offset:offset+size],global_wire_specs(length,H//2)));offset+=size
        return base.join_received(parts,'fp4_global')

    def execute(self,dense,arm):
        if arm=='fp4_table16':return super().execute(dense,arm)
        assert arm=='fp4_global'
        import torch
        q,k,v=dense;all_c,all_mean=self.global_means(q,k)
        qc=(q-all_c).contiguous();kc=k-all_mean;kc[:,VALID[self.rank]:]=0
        q4,qs=self.pack(qc,'q');k4,ks=self.pack(kc,'k');v4t,vsft=self.pack(v,'v')
        local_table=self.table(all_c,kc)
        full=self.global_forward_exchange(dict(q4=q4,qs=qs,k4=k4,ks=ks,v4t=v4t,vsft=vsft,table=local_table))
        q4,qs,k4,ks,v4t,vsft,table=(full[name] for name in ('q4','qs','k4','ks','v4t','vsft','table'))
        head_slice=slice(self.rank*(H//2),(self.rank+1)*(H//2))
        c=all_c[head_slice].contiguous();mean=all_mean[head_slice].contiguous()
        del full,all_c,all_mean,qc,kc,local_table
        self.budget();self.counts['attention']+=1
        output=self.official.nvfp4_attention_sm120_fwd(*(t.unsqueeze(0) for t in (q4,k4,v4t,qs,ks,vsft)),
            table.unsqueeze(0),sm_scale=D**-.5,causal=False,per_block_mean=False,
            out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)[0]
        wire,send_sizes=base.encode_inverse(output)
        part_bytes=(H//2)*LENGTHS[self.rank]*D*2
        received=self.a2a(wire,send_sizes,[part_bytes,part_bytes],'inverse_BF16_output')
        restored=base.decode_inverse(received,self.rank,H)
        return dict(output=restored[:,:VALID[self.rank]].transpose(0,1).contiguous(),centers=c,k_mean=mean)


def binding(report,check_data):
    import torch
    ids=base.binding(report,check_data)
    prior_path=ROOT/'results/research/E036/run_rank0.json';prior=json.loads(prior_path.read_text())
    assert prior['status']=='complete' and prior['actual_counts']['attention']==39
    assert common.record(base.__file__)==prior['sources']['runner']
    peer_path=prior_path.with_name('run_rank1.json');peer=json.loads(peer_path.read_text())
    assert peer['status']=='complete' and peer['actual_counts']['attention']==39
    report['sources'].update(runner=common.record(__file__),sequence_base=common.record(base.__file__),plan=common.record(PLAN))
    old=json.loads((ROOT/'results/research/E034/run.json').read_text())
    global_ref=next(row['output'] for row in old['benchmark'] if row['block']==0 and row['arm']=='global')
    report['references']['global']=global_ref
    report['e036_reports']=[common.record(path) for path in (prior_path,peer_path)]
    report['geometry']['global_contract']=dict(padded_q_denominator=NP,valid_k_denominator=N,
        table_rows=1,mean_input='Resident BF16 local operands, FP32 sums in one AllReduce, divide then BF16 cast.',
        q_padding='Zeros included in padded-Q denominator; centered Q padding equals -C.',
        k_padding='Padding excluded from K sum; centered K padding set back to zero.')
    if check_data:
        value=common.load(global_ref['artifact']);assert value.shape==(N,H,D) and value.dtype==torch.bfloat16
    return ids


def cpu_check(report):
    import torch
    # E036 has already checked and executed the unchanged physical transport.
    old=json.loads((ROOT/'results/research/E036/check.json').read_text())
    assert old['status']=='complete' and old['cpu_transport']['v_scale_physical_reference_rejoin']
    report['inherited_transport_check']=common.record(ROOT/'results/research/E036/check.json')
    heads=4;half=2;fields_by_source=[];sent=[]
    for rank,length in enumerate(LENGTHS):
        fields={}
        for field_index,(name,shape,dtype) in enumerate(global_wire_specs(length,heads)):
            nbytes=math.prod(shape)*torch.empty((),dtype=dtype).element_size()
            labels=((torch.arange(nbytes,dtype=torch.int64)+rank*71+field_index*37)%251).to(torch.uint8)
            fields[name]=labels.view(dtype).reshape(shape)
        fields_by_source.append(fields);sent.append(base.encode_head_payload(fields,heads))
    for destination in range(2):
        parts=[]
        for source,(wire,sizes) in enumerate(sent):
            start=sum(sizes[:destination]);part=wire[start:start+sizes[destination]].clone()
            parts.append(base.decode_payload(part,global_wire_specs(LENGTHS[source],half)))
        joined=base.join_received(parts,'fp4_global')
        expected=base.join_received(fields_by_source,'fp4_global')
        for name in joined:assert base.equal_bytes(joined[name],expected[name][destination*half:(destination+1)*half])
        assert joined['table'].shape==(half,1,NP)
    # Toy exact-count fixture checks different denominators and padding, not rounding parity.
    sums=[]
    for rank,length in enumerate(LENGTHS):
        q=torch.zeros((2,length,1),dtype=torch.bfloat16);q[:,:VALID[rank]]=1
        k=torch.ones_like(q);k[:,VALID[rank]:]=9
        sums.append(local_qk_sums(q,k,rank))
    qmean,kmean=means_from_sums(sums[0]+sums[1])
    assert bool((qmean==torch.tensor(N/NP,dtype=torch.bfloat16)).all()) and bool((kmean==1).all())
    report['cpu_global_check']=dict(labeled_T1_payload_and_columns=True,correct_distinct_QK_denominators=True,
        combined_reduction_shape=[2,H,1,D],combined_reduction_bytes=2*H*D*4,
        scope='Only new T1 payload and padded-Q/valid-K mean bookkeeping. Inherited E036 physical scale/inverse mapping unchanged. No GPU initialization or native numerical comparison.')


def run(args,report,ids_cpu,budget):
    import torch
    import torch.distributed as dist
    import flashinfer.nvfp4_attention_sm120 as official
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    assert rank==local_rank and int(os.environ['WORLD_SIZE'])==2
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    torch.cuda.set_device(local_rank);assert torch.cuda.get_device_capability()==(12,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(local_rank).total_memory))
    dist.init_process_group('nccl',timeout=timedelta(seconds=max(1,args.deadline_unix-time.time())),device_id=torch.device('cuda',local_rank))
    report.update(rank=rank,world_size=2,local_rank=local_rank,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=list(torch.cuda.nccl.version()),
            device=torch.cuda.get_device_name(),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],tf32=False,
            p2p_access_to_other_rank=bool(torch.cuda.can_device_access_peer(local_rank,1-local_rank)),cpu_threads=torch.get_num_threads()))
    checked=json.loads((RD/'check.json').read_text());assert checked['status']=='complete'
    for key in ('sources','reference_report','input','references','geometry','private_source_hash','e036_reports'):
        assert checked[key]==report[key],key
    rank_dir=args.data/f'rank{rank}';assert not rank_dir.exists();rank_dir.mkdir(parents=True)
    source=Path(report['input']['file']);assert source.stat().st_size==report['input']['bytes']
    raw=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    dense=tuple(torch.nn.functional.pad(raw[name][STARTS[rank]:STARTS[rank]+VALID[rank]].transpose(0,1),
        (0,0,0,LENGTHS[rank]-VALID[rank])).contiguous().cuda() for name in ('q','k','v'))
    del raw
    ids=ids_cpu[:,:H//2].contiguous().cuda();base.private.validate_center_ids(ids,base.K)
    module=official.get_nvfp4_attention_sm120_module();base.private.get_codebook_module();budget()
    engine=Execution(rank,module,ids,budget,official)
    report['actual_counts']=engine.counts;report['control_barrier_calls']=0
    def synchronize_boundary():
        budget();dist.barrier();report['control_barrier_calls']+=1;torch.cuda.synchronize()
    def memory():
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    synchronize_boundary();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    report['resident_baseline']=memory()
    entries={arm:dict(arm=arm,warmups=[],repeats=[]) for arm in ARMS};report['benchmarks']=list(entries.values())
    def invoke(arm,round_index,order,position,measured):
        synchronize_boundary();torch.cuda.reset_peak_memory_stats()
        before=memory();old=dict(engine.counts);engine.collectives=[]
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        started=time.perf_counter();begin.record();products=engine.execute(dense,arm)
        end.record();torch.cuda.synchronize();wall=(time.perf_counter()-started)*1000
        after=memory();budget();assert after['peak_allocated']<=60*2**30
        receipt=dict(round=round_index,order=list(order),position=position,wall_ms=wall,cuda_ms=begin.elapsed_time(end),
            before=before,after=after,incremental_peak_allocated=after['peak_allocated']-before['allocated'],
            incremental_peak_reserved=after['peak_reserved']-before['reserved'],
            actual_counts={key:engine.counts[key]-old[key] for key in old},collectives=engine.collectives)
        entries[arm]['repeats' if measured else 'warmups'].append(receipt)
        if measured and round_index==9:
            output=products['output'].cpu();state={name:products[name].cpu() for name in ('centers','k_mean')}
            assert bool(output.isfinite().all()) and all(bool(v.isfinite().all()) for v in state.values())
            path=rank_dir/f'{arm}_output.pt';torch.save(output,path)
            state_path=rank_dir/f'{arm}_state.pt';torch.save(state,state_path)
            entries[arm]['output']=dict(artifact=common.record(path),tensor=dict(shape=list(output.shape),dtype=str(output.dtype),finite=True))
            entries[arm]['state']=dict(artifact=common.record(state_path),tensors={name:common.trecord(value) for name,value in state.items()},
                ownership='C/Kmean own this rank28heads; output owns original valid token range/all56heads. C has16rows for coarse and1 for global.')
        del products;common.save(args.output,report)
    for warm in range(3):
        for position,arm in enumerate(ARMS):invoke(arm,warm,ARMS,position,False)
    report['after_warmup_baseline']=memory()
    for repeat in range(10):
        order=ARMS if repeat%2==0 else ARMS[::-1]
        for position,arm in enumerate(order):invoke(arm,repeat,order,position,True)
    assert engine.counts['attention']==26 and engine.counts['all_to_all']==65 and engine.counts['all_reduce']==26
    assert engine.counts['q_sum_ops']==13 and engine.counts['k_sum_ops']==26
    assert all(engine.counts[name]==26 for name in ('q_pack','k_pack','v_pack','correction_gemm'))
    for entry in entries.values():
        entry.update(status='complete',median_wall_ms=statistics.median(r['wall_ms'] for r in entry['repeats']),
            median_cuda_ms=statistics.median(r['cuda_ms'] for r in entry['repeats']))
    synchronize_boundary();report.update(status='complete',actual_dit_calls=0,cuda_initialized=True)
    del dense,ids,engine,module;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    report['released_allocated_bytes']=torch.cuda.memory_allocated();dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--deadline-unix',type=float);parser.add_argument('--output',type=Path)
    parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args();rank=int(os.environ.get('RANK','0'))
    args.output=args.output or RD/('check.json' if args.phase=='check' else f'run_rank{rank}.json')
    assert not args.output.exists(), 'Preserve prior result/failed attempt'
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=900
    else:assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E037 original deadline')
    report=dict(experiment='E037',phase=args.phase,status='running',rank=rank,world_size=2,
        arms=list(ARMS),expected_native_per_rank=26,expected_native_total=52,actual_dit_calls=0,
        timing_scope='Same E036 complete token→head→token boundary. Include source means, every data collective, packs, T, wire copies/reassembly, native attention and inverse output. Entry control barriers, JIT/upload and saving excluded; cold JIT remains in900s deadline.',
        memory_scope='Shared allocator warmed across both arms, no per-repeat cache flush; preserve each baseline/peak. Release unused construction tensors before attention and products before next call.',
        byte_scope='Actual A2A application payload, self-copy/receive separated. AllReduce logical tensor payload recorded separately; no claimed measurement of NCCL link/protocol bytes.',
        scientific_scope='Global-Q/T1 is a different accuracy contract and strong baseline, not an equal-quality speedup. Compare each arm to its E034 counterpart and common BF16; no cross-arm byte gate.')
    try:
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        ids=binding(report,args.phase=='check')
        if args.phase=='check':
            cpu_check(report);assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            with torch.inference_mode():run(args,report,ids,budget)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(rank=rank,status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
