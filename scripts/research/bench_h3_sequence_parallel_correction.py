#!/usr/bin/env python3
"""E036: two-rank complete attention boundary, fixed coarse16 correction.

CPU check performs labeled byte transport, not distributed/GPU execution.
Run uses NCCL split-size all_to_all_single and the unchanged E033 consumer.
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

import probe_h3_query_mean_k4 as common
import bench_h3_coarse_centers as coarse
import codebook_attention_sm120 as private

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E036'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E036')
PLAN=ROOT/'research_state/06_experiments/E036_sequence_parallel_correction_plan.md'
N,NP,H,D,G,K=22539,22656,56,128,177,16
LENGTHS=(11264,11392)
VALID=(11264,11275)
STARTS=(0,11264)
ARMS=('bf16_a2a','mixed_k16','fp4_table16')
COUNTERS=('q_center_reduce_ops','k_mean_ops','k_sum_ops','q_pack','k_pack','v_pack',
          'correction_gemm','attention','all_to_all','all_reduce')


def wire_specs(arm,length,heads):
    import torch
    dense=(heads,length,D); q=(heads,length,D//2); sf=(heads,length,D//16)
    vt=(heads,D,length//2); vs=(heads,D,length//16)
    if arm=='bf16_a2a':return [(name,dense,torch.bfloat16) for name in ('q','k','v')]
    rows=[('q4',q,torch.uint8),('qs',sf,torch.float8_e4m3fn)]
    if arm=='mixed_k16':rows.append(('k',dense,torch.bfloat16))
    else:rows.extend([('k4',q,torch.uint8),('ks',sf,torch.float8_e4m3fn)])
    rows.extend([('v4t',vt,torch.uint8),('vsft',vs,torch.float8_e4m3fn)])
    rows.append(('centers',(heads,8,D),torch.bfloat16) if arm=='mixed_k16'
                else ('table',(heads,K,length),torch.float32))
    return rows


def spec_bytes(spec):
    import torch
    return sum(math.prod(shape)*torch.empty((),dtype=dtype).element_size() for _,shape,dtype in spec)


def encode_head_payload(fields,heads):
    """One wire allocation, destination-major, with all fields coalesced."""
    import torch
    half=heads//2; per_destination=sum(t.numel()*t.element_size()//2 for t in fields.values())
    device=next(iter(fields.values())).device
    wire=torch.empty(2*per_destination,dtype=torch.uint8,device=device)
    for destination in range(2):
        offset=destination*per_destination
        for value in fields.values():
            part=value[destination*half:(destination+1)*half].contiguous().view(torch.uint8).reshape(-1)
            wire[offset:offset+part.numel()].copy_(part);offset+=part.numel()
        assert offset==(destination+1)*per_destination
    return wire,[per_destination,per_destination]


def decode_payload(wire,spec):
    offset=0;values={}
    for name,shape,dtype in spec:
        import torch
        size=math.prod(shape)*torch.empty((),dtype=dtype).element_size()
        values[name]=wire[offset:offset+size].view(dtype).reshape(shape)
        offset+=size
    assert offset==wire.numel()
    return values


def join_received(parts,arm):
    import torch
    result={}
    for name in parts[0]:
        if name=='vsft':
            heads=parts[0][name].shape[0]
            tiles=[part[name].view(torch.uint8).reshape(heads,D//64,length//64,256)
                   for part,length in zip(parts,LENGTHS)]
            result[name]=torch.cat(tiles,dim=2).reshape(heads,D,NP//16).view(torch.float8_e4m3fn)
        else:
            dim=2 if name in ('v4t','table') else 1
            result[name]=torch.cat([part[name] for part in parts],dim=dim).contiguous()
    return result


def encode_inverse(output):
    import torch
    pieces=[output[:,begin:begin+length].contiguous().view(torch.uint8).reshape(-1)
            for begin,length in zip(STARTS,LENGTHS)]
    return torch.cat(pieces),[piece.numel() for piece in pieces]


def decode_inverse(wire,rank,heads):
    import torch
    size=heads//2*LENGTHS[rank]*D*2
    pieces=[wire[i*size:(i+1)*size].view(torch.bfloat16).reshape(heads//2,LENGTHS[rank],D) for i in range(2)]
    return torch.cat(pieces,dim=0).contiguous()


def equal_bytes(a,b):
    import torch
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def binding(report,check_data):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    prior_path=ROOT/'results/research/E034/run.json';prior=json.loads(prior_path.read_text())
    assert prior['status']=='complete'
    source=next(row for row in prior['inputs'] if row['block']==0)
    target=next(row for row in prior['benchmark'] if row['block']==0 and row['arm']=='coarse16')
    for key,path in (('private',private.__file__),('official',official.__file__),('runner',coarse.__file__)):
        assert common.record(path)==prior['sources'][key],key
    report.update(sources=dict(runner=common.record(__file__),common=common.record(common.__file__),
        coarse=common.record(coarse.__file__),private=common.record(private.__file__),official=common.record(official.__file__),
        plan=common.record(PLAN)),reference_report=common.record(prior_path),
        input=source['capture'],references=dict(bf16=source['outputs']['bf16'],coarse16=target['output'],
            coarse_centers=target['centers']),private_source_hash=prior['private_source']['source_hash'])
    assert tuple(coarse.BOUNDARIES)==tuple(j*G//K for j in range(K+1))
    assert coarse.BOUNDARIES[8]*128==LENGTHS[0]
    assert sum(LENGTHS)==NP and sum(VALID)==N and all(length%128==0 for length in LENGTHS)
    ids=coarse.coarse_ids_cpu()
    private.validate_center_ids(ids,K)
    report['geometry']=dict(valid_length=N,padded_length=NP,heads=H,head_dim=D,world_size=2,
        token_starts=list(STARTS),padded_token_lengths=list(LENGTHS),valid_token_lengths=list(VALID),
        head_ranges=[[0,28],[28,56]],center_boundaries=list(coarse.BOUNDARIES),ids=common.trecord(ids),
        q_padding='Zero raw Q participates in C, then Qcenter padding is -C.',
        k_padding='Kmean uses valid N; Kcenter padding is explicitly zero, never -Kmean.',
        scale_transport='Q/K row64 slabs; Vsf reshaped [H,2,L/64,256], concatenate token-tile axis.')
    if check_data:
        raw=common.load(source['capture'])
        for name in ('q','k','v'):
            assert raw[name].shape==(N,H,D) and raw[name].dtype==torch.bfloat16
            assert bool(raw[name].isfinite().all())
        assert raw['scale']==D**-.5 and raw['valid_length']==N
        for ref in (source['outputs']['bf16'],target['output']):
            value=common.load(ref['artifact']);assert value.shape==(N,H,D) and value.dtype==torch.bfloat16
        saved=common.load(target['centers']['artifact'])
        assert saved['centers'].shape==(H,K,D) and torch.equal(saved['ids'],ids)
    return ids


def cpu_transport_check(report):
    import torch
    # Actual unequal lengths/D and physical tile dimensions, with four labeled heads.
    heads=4;half=2;proof=[]
    for arm_index,arm in enumerate(ARMS):
        shards=[];sent=[]
        for rank,length in enumerate(LENGTHS):
            fields={}
            for field_index,(name,shape,dtype) in enumerate(wire_specs(arm,length,heads)):
                nbytes=math.prod(shape)*torch.empty((),dtype=dtype).element_size()
                labels=((torch.arange(nbytes,dtype=torch.int64)+rank*71+field_index*37+arm_index*19)%251).to(torch.uint8)
                fields[name]=labels.view(dtype).reshape(shape)
            shards.append(fields);sent.append(encode_head_payload(fields,heads))
        for destination in range(2):
            received=[]
            for source in range(2):
                wire,sizes=sent[source];begin=sum(sizes[:destination])
                received.append(wire[begin:begin+sizes[destination]].clone())
            parts=[decode_payload(value,wire_specs(arm,length,half)) for value,length in zip(received,LENGTHS)]
            joined=join_received(parts,arm)
            expected=join_received(shards,arm)
            for name in joined:assert equal_bytes(joined[name],expected[name][destination*half:(destination+1)*half])
        proof.append(dict(arm=arm,head_redistribution_and_physical_packet_rejoin=True,
            send_bytes_per_source=[s[1] for s in sent]))
    # Inverse output: unequal sequence splits, full head order restored.
    full=((torch.arange(heads*NP*D*2,dtype=torch.int64)%251).to(torch.uint8).view(torch.bfloat16).reshape(heads,NP,D))
    sent=[encode_inverse(full[rank*half:(rank+1)*half]) for rank in range(2)]
    for destination in range(2):
        chunks=[]
        for wire,sizes in sent:
            begin=sum(sizes[:destination]);chunks.append(wire[begin:begin+sizes[destination]])
        restored=decode_inverse(torch.cat(chunks),destination,heads)
        assert equal_bytes(restored,full[:,STARTS[destination]:STARTS[destination]+LENGTHS[destination]])
    # Independent full physical Vsf reference, not two calls to the join helper.
    full_sf=(torch.arange(heads*D*(NP//16),dtype=torch.int64)%251).to(torch.uint8).reshape(heads,D,NP//16)
    physical=full_sf.reshape(heads,D//64,NP//64,256)
    pieces=[dict(vsft=physical[:,:,start//64:(start+length)//64].contiguous()
                 .reshape(heads,D,length//16).view(torch.float8_e4m3fn)) for start,length in zip(STARTS,LENGTHS)]
    assert equal_bytes(join_received(pieces,'fp4_table16')['vsft'],full_sf.view(torch.float8_e4m3fn))
    # Distinct row labels additionally check K's local permutation phase.
    def perm(t):
        u=t%32;return (t//32)*32+(u//8)*2+((u%8)//2)*8+u%2
    assert torch.equal(perm(torch.arange(LENGTHS[1]))+STARTS[1],perm(torch.arange(STARTS[1],NP)))
    packets=[torch.empty(shape,dtype=dtype,device='meta') for shape,dtype in (
        ((1,28,NP,64),torch.uint8),((1,28,NP,64),torch.uint8),((1,28,D,NP//2),torch.uint8),
        ((1,28,NP,8),torch.float8_e4m3fn),((1,28,NP,8),torch.float8_e4m3fn),((1,28,D,NP//16),torch.float8_e4m3fn))]
    private.check_metadata(*packets,torch.empty((1,28,K,NP),dtype=torch.float32,device='meta'),
        torch.empty((1,28,G),dtype=torch.int32,device='meta'),unpadded_k_len=N,require_cuda=False)
    report['cpu_transport']=dict(forward=proof,inverse_output_roundtrip=True,k_permutation_phase=True,v_scale_physical_reference_rejoin=True,
        private_28head_metadata=True,scope='Labeled byte copies at actual token lengths and D128, reduced four-head fixture. No GPU packet arithmetic, distributed collective, or native28head execution validated.')


class Execution:
    def __init__(self,rank,module,ids,budget):
        self.rank=rank;self.module=module;self.ids=ids;self.budget=budget
        self.counts={key:0 for key in COUNTERS};self.collectives=[]

    def a2a(self,wire,send_sizes,recv_sizes,label,fields=None):
        import torch
        import torch.distributed as dist
        self.budget();out=torch.empty(sum(recv_sizes),dtype=torch.uint8,device=wire.device)
        self.counts['all_to_all']+=1
        self.collectives.append(dict(kind='all_to_all_single',label=label,dtype=str(wire.dtype),
            input_numel=wire.numel(),output_numel=out.numel(),element_size=1,
            send_splits=send_sizes,recv_splits=recv_sizes,remote_send_bytes=send_sizes[1-self.rank],
            remote_receive_bytes=recv_sizes[1-self.rank],self_send_bytes=send_sizes[self.rank],
            self_receive_bytes=recv_sizes[self.rank],fields=fields,
            byte_scope='Actual application split payload; NCCL protocol/transport overhead is not measured.'))
        dist.all_to_all_single(out,wire,output_split_sizes=recv_sizes,input_split_sizes=send_sizes)
        return out

    def forward_exchange(self,fields,arm):
        wire,send_sizes=encode_head_payload(fields,H)
        sizes=[spec_bytes(wire_specs(arm,length,H//2)) for length in LENGTHS]
        desc={name:dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size()) for name,value in fields.items()}
        received=self.a2a(wire,send_sizes,sizes,'forward_'+arm,desc)
        parts=[];offset=0
        for size,length in zip(sizes,LENGTHS):
            parts.append(decode_payload(received[offset:offset+size],wire_specs(arm,length,H//2)));offset+=size
        return join_received(parts,arm)

    def centers(self,q,local):
        import torch
        start=STARTS[self.rank] if local else 0
        first,last=(self.rank*8,(self.rank+1)*8) if local else (0,K)
        self.counts['q_center_reduce_ops']+=last-first
        return torch.stack([q[:,128*coarse.BOUNDARIES[j]-start:128*coarse.BOUNDARIES[j+1]-start].mean(1)
                            for j in range(first,last)],dim=1)

    def qcenter(self,q,c,local):
        start=STARTS[self.rank]//128 if local else 0
        groups=q.shape[1]//128
        indices=self.ids[0,0,start:start+groups].long()-(self.rank*8 if local else 0)
        selected=c.index_select(1,indices)
        return (q.reshape(q.shape[0],groups,128,D)-selected.unsqueeze(2)).reshape_as(q).contiguous()

    def pack(self,value,kind):
        import torch
        heads,length,_=value.shape
        if kind=='v':
            codes=torch.empty((1,heads,D,length//2),dtype=torch.uint8,device=value.device)
            scales=torch.empty((1,heads,D,length//16),dtype=torch.float8_e4m3fn,device=value.device)
            self.module.scaled_fp4_quant_trans(value.unsqueeze(0),codes,scales,1)
        else:
            codes=torch.empty((1,heads,length,D//2),dtype=torch.uint8,device=value.device)
            scales=torch.empty((1,heads,length,D//16),dtype=torch.float8_e4m3fn,device=value.device)
            fn=self.module.scaled_fp4_quant if kind=='q' else self.module.scaled_fp4_quant_permute
            fn(value.unsqueeze(0),codes,scales,1)
        self.counts[kind+'_pack']+=1
        return codes[0],scales[0]

    def owner_key(self,k):
        self.counts['k_mean_ops']+=1
        mean=k[:,:N].mean(1,keepdim=True)
        centered=k-mean;centered[:,N:]=0
        return centered,mean

    def table(self,c,kc):
        self.counts['correction_gemm']+=1
        return (c.float()@kc.transpose(-2,-1).float()).contiguous()

    def exchange_centers(self,c):
        import torch
        raw=c.contiguous().view(torch.uint8).reshape(-1)
        wire=torch.cat((raw,raw));sizes=[raw.numel(),raw.numel()]
        received=self.a2a(wire,sizes,sizes,'centers_all_heads',dict(centers=dict(shape=list(c.shape),dtype=str(c.dtype))))
        pieces=[received[i*sizes[0]:(i+1)*sizes[0]].view(torch.bfloat16).reshape(H,8,D) for i in range(2)]
        return torch.cat(pieces,dim=1).contiguous()

    def global_key_mean(self,k):
        import torch
        import torch.distributed as dist
        self.counts['k_sum_ops']+=1
        value=k[:,:VALID[self.rank]].sum(1,keepdim=True,dtype=torch.float32)
        self.counts['all_reduce']+=1
        size=value.numel()*value.element_size()
        self.collectives.append(dict(kind='all_reduce_sum',label='valid_K_sum',dtype=str(value.dtype),
            input_numel=value.numel(),output_numel=value.numel(),element_size=value.element_size(),
            logical_peer_send_bytes=size,logical_peer_receive_bytes=size,self_send_bytes=0,
            byte_scope='Actual FP32 reduction tensor; world2 one-peer data-payload accounting, not a measurement of NCCL algorithm/protocol bytes.'))
        dist.all_reduce(value,op=dist.ReduceOp.SUM)
        return (value/N).to(torch.bfloat16)

    def execute(self,dense,arm):
        import torch
        q,k,v=dense;head_slice=slice(self.rank*(H//2),(self.rank+1)*(H//2))
        if arm=='bf16_a2a':
            full=self.forward_exchange(dict(q=q,k=k,v=v),arm)
            c=self.centers(full['q'],False);qc=self.qcenter(full['q'],c,False)
            kc,mean=self.owner_key(full['k'])
            q4,qs=self.pack(qc,'q');k4,ks=self.pack(kc,'k');v4t,vsft=self.pack(full['v'],'v')
            table=self.table(c,kc)
            del full,qc,kc
        elif arm=='mixed_k16':
            local_c=self.centers(q,True);qc=self.qcenter(q,local_c,True)
            q4,qs=self.pack(qc,'q');v4t,vsft=self.pack(v,'v')
            full=self.forward_exchange(dict(q4=q4,qs=qs,k=k,v4t=v4t,vsft=vsft,centers=local_c),arm)
            c=full['centers'];kc,mean=self.owner_key(full['k']);k4,ks=self.pack(kc,'k')
            q4,qs,v4t,vsft=(full[name] for name in ('q4','qs','v4t','vsft'))
            table=self.table(c,kc)
            del full,local_c,qc,kc
        else:
            assert arm=='fp4_table16'
            local_c=self.centers(q,True);all_c=self.exchange_centers(local_c)
            all_mean=self.global_key_mean(k)
            qc=self.qcenter(q,local_c,True);kc=k-all_mean;kc[:,VALID[self.rank]:]=0
            q4,qs=self.pack(qc,'q');k4,ks=self.pack(kc,'k');v4t,vsft=self.pack(v,'v')
            local_table=self.table(all_c,kc)
            full=self.forward_exchange(dict(q4=q4,qs=qs,k4=k4,ks=ks,v4t=v4t,vsft=vsft,table=local_table),arm)
            q4,qs,k4,ks,v4t,vsft,table=(full[name] for name in ('q4','qs','k4','ks','v4t','vsft','table'))
            c=all_c[head_slice].contiguous();mean=all_mean[head_slice].contiguous()
            del full,local_c,all_c,all_mean,qc,kc,local_table
        self.budget();self.counts['attention']+=1
        output=private.codebook_fwd(*(t.unsqueeze(0) for t in (q4,k4,v4t,qs,ks,vsft)),
            table.unsqueeze(0),self.ids,sm_scale=D**-.5,unpadded_k_len=N)[0]
        wire,send_sizes=encode_inverse(output)
        part_bytes=(H//2)*LENGTHS[self.rank]*D*2
        received=self.a2a(wire,send_sizes,[part_bytes,part_bytes],'inverse_BF16_output')
        restored=decode_inverse(received,self.rank,H)
        valid_output=restored[:,:VALID[self.rank]].transpose(0,1).contiguous()
        return dict(output=valid_output,centers=c,k_mean=mean)


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
            device=torch.cuda.get_device_name(),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],tf32=False))
    report['environment']['p2p_access_to_other_rank']=bool(torch.cuda.can_device_access_peer(local_rank,1-local_rank))
    report['environment']['cpu_threads']=torch.get_num_threads()
    checked=json.loads((RD/'check.json').read_text());assert checked['status']=='complete'
    for key in ('sources','reference_report','input','references','geometry','private_source_hash'):assert checked[key]==report[key],key
    rank_dir=args.data/f'rank{rank}';assert not rank_dir.exists();rank_dir.mkdir(parents=True)
    # CPU check already verified the large input file; run maps it without a second full hash.
    source=Path(report['input']['file']);assert source.stat().st_size==report['input']['bytes']
    raw=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    dense=tuple(torch.nn.functional.pad(raw[name][STARTS[rank]:STARTS[rank]+VALID[rank]].transpose(0,1),
        (0,0,0,LENGTHS[rank]-VALID[rank])).contiguous().cuda() for name in ('q','k','v'))
    del raw
    ids=ids_cpu[:,:H//2].contiguous().cuda();private.validate_center_ids(ids,K)
    module=official.get_nvfp4_attention_sm120_module();private.get_codebook_module();budget()
    engine=Execution(rank,module,ids,budget)
    report['actual_counts']=engine.counts;report['control_barrier_calls']=0
    def synchronize_boundary():
        budget();dist.barrier();report['control_barrier_calls']+=1;torch.cuda.synchronize()
    def memory():
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    synchronize_boundary();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    report['resident_baseline']=memory()
    entries={arm:dict(arm=arm,warmups=[],repeats=[]) for arm in ARMS}
    report['benchmarks']=list(entries.values())
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
                ownership='C and Kmean belong to this rank head range; output belongs to original valid token range and all56heads.')
        del products
        common.save(args.output,report)
    for warm in range(3):
        for position,arm in enumerate(ARMS):invoke(arm,warm,ARMS,position,False)
    report['after_warmup_baseline']=memory()
    for repeat in range(10):
        shift=repeat%3;order=ARMS[shift:]+ARMS[:shift]
        for position,arm in enumerate(order):invoke(arm,repeat,order,position,True)
    assert engine.counts['attention']==39 and engine.counts['all_to_all']==91 and engine.counts['all_reduce']==13
    assert all(engine.counts[name]==39 for name in ('q_pack','k_pack','v_pack','correction_gemm'))
    for entry in entries.values():
        entry.update(status='complete',median_wall_ms=statistics.median(r['wall_ms'] for r in entry['repeats']),
            median_cuda_ms=statistics.median(r['cuda_ms'] for r in entry['repeats']))
    synchronize_boundary();report.update(status='complete',actual_dit_calls=0,cuda_initialized=True)
    del dense,ids,engine,module;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    report['released_allocated_bytes']=torch.cuda.memory_allocated()
    dist.destroy_process_group()


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
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E036 original deadline')
    report=dict(experiment='E036',phase=args.phase,status='running',rank=rank,world_size=2,
        arms=list(ARMS),expected_native_per_rank=39,expected_native_total=78,actual_dit_calls=0,
        timing_scope='Resident padded token-shard BF16 HND QKV to valid token-owned/all-head NHD BF16 output; includes every data collective, C/K statistics, pack, wire copies/reassembly, correction and output inverse A2A. Control barriers/sync before start, JIT/upload, saving and CPU numeric work excluded.',
        memory_scope='Shared allocator warmed across all arms; no per-repeat cache flush. Resident/after-warmup and each repeat baseline/peaks retained. Outputs released before next call.',
        byte_scope='A2A actual dtype/numel/splits and peer payload separated from self-copy/receive. AllReduce tensor payload reported separately, not claimed as measured NCCL link/protocol traffic. No performance or video-quality inference from payload ratios.',
        cold_jit_in_deadline=True)
    try:
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        ids=binding(report,args.phase=='check')
        if args.phase=='check':
            cpu_transport_check(report);assert not torch.cuda.is_initialized()
            report.update(status='complete',cuda_initialized=False)
        else:
            with torch.inference_mode():run(args,report,ids,budget)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(rank=rank,status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
