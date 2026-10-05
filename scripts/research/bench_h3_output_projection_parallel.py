#!/usr/bin/env python3
"""E039: head-owned attention output to token-owned native SVD projection.

Four fixed communication/execution contracts, 52 boundaries / 65 FP4 GEMMs
per rank, 0 DiT. Reuses the frozen H3 quantizer and native GEMM implementation.
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

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E039'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E039')
PLAN=ROOT/'research_state/06_experiments/E039_output_projection_parallel_plan.md'
INPUT=DATA/'attention_output.pt'
WEIGHT=Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export/layers/blocks.0.attn.out_proj.pt')
N,REAL_TOTAL,H,D,DIN,DOUT,R=22539,22592,56,128,7168,5376,32
STARTS=(0,11264)
REAL=(11264,11328)
VALID=(11264,11275)
PHYSICAL=(11264,11392)
SLOT=max(PHYSICAL)
HALF=DIN//2
ARMS=('bf16_return','fp8_return','row_parallel','fp4_side')
COUNTERS=('boundary','native_gemm','lr_down','lr_up','smooth','pack','fp8_quant','fp8_dequant',
          'amax_reduce','all_to_all','all_reduce','reduce_scatter')


def dtype_bytes(dtype):
    import torch
    return torch.empty((),dtype=dtype).element_size()


def specs(arm,destination):
    import torch
    m=REAL[destination]
    if arm=='bf16_return':return [('o',(m,HALF),torch.bfloat16)]
    if arm=='fp8_return':return [('codes',(m,HALF),torch.uint8),('scale',(1,),torch.float32)]
    assert arm=='fp4_side'
    return [('codes',(m,HALF//2),torch.uint8),
            ('sf',(PHYSICAL[destination]*(HALF//16),),torch.float8_e4m3fn),
            ('down',(m,R),torch.bfloat16)]


def nbytes(spec):
    return sum(math.prod(shape)*dtype_bytes(dtype) for _,shape,dtype in spec)


def encode(parts):
    import torch
    sizes=[sum(v.numel()*v.element_size() for v in part.values()) for part in parts]
    first=next(iter(parts[0].values()))
    wire=torch.empty(sum(sizes),device=first.device,dtype=torch.uint8)
    offset=0
    for part in parts:
        for value in part.values():
            raw=value.contiguous().view(torch.uint8).reshape(-1)
            wire[offset:offset+raw.numel()].copy_(raw);offset+=raw.numel()
    return wire,sizes


def decode(wire,spec):
    result={};offset=0
    for name,shape,dtype in spec:
        count=math.prod(shape)*dtype_bytes(dtype)
        result[name]=wire[offset:offset+count].view(dtype).reshape(shape)
        offset+=count
    assert offset==wire.numel()
    return result


def join_activation(parts,destination):
    """Logical K halves concatenate codes; SF concatenates physical column tiles."""
    import torch
    rows=PHYSICAL[destination]//128
    sf=torch.cat([p['sf'].view(rows,(HALF//16)//4,512) for p in parts],dim=1).contiguous().view(-1)
    codes=torch.cat([p['codes'] for p in parts],dim=1).contiguous()
    return codes,sf


def half_weight(payload,rank):
    import torch
    from wan_native_nvfp4 import swizzle_scales,unswizzle_scales
    from h3_native_nvfp4 import NativeH3Linear,PackedNVFP4
    t=payload['tensors'];lo=rank*HALF;hi=lo+HALF
    codes=t['weight_packed'][:,lo//2:hi//2].contiguous()
    scales=unswizzle_scales(t['weight_scales_swizzled'],DOUT,DIN//16)[:,lo//16:hi//16].contiguous()
    packet=PackedNVFP4(codes,None,t['weight_global'],swizzle_scales(scales),(DOUT,HALF),'original_E009_K_half')
    return NativeH3Linear(packet,t['smooth'][lo:hi].contiguous(),t['lr_a'][:,lo:hi].contiguous(),t['lr_b'],None)


class Execution:
    def __init__(self,rank,full,half,budget):
        self.rank=rank;self.full=full;self.half=half;self.budget=budget
        self.counts={name:0 for name in COUNTERS};self.collectives=[];self.flags=[]

    def a2a(self,parts,arm):
        import torch
        import torch.distributed as dist
        wire,sizes=encode(parts);receive_sizes=[nbytes(specs(arm,self.rank))]*2
        received=torch.empty(sum(receive_sizes),device=wire.device,dtype=torch.uint8)
        self.counts['all_to_all']+=1
        self.collectives.append(dict(kind='all_to_all',label=arm,dtype='torch.uint8',
            send_splits=sizes,receive_splits=receive_sizes,input_numel=wire.numel(),output_numel=received.numel(),
            remote_send_bytes=sizes[1-self.rank],remote_receive_bytes=receive_sizes[1-self.rank],
            self_send_bytes=sizes[self.rank],fields=[[
                dict(name=k,shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size())
                for k,v in p.items()] for p in parts]))
        self.budget();dist.all_to_all_single(received,wire,receive_sizes,sizes)
        size=receive_sizes[0]
        return [decode(received[i*size:(i+1)*size],specs(arm,self.rank)) for i in range(2)]

    def destination_globals(self,x):
        import torch
        import torch.distributed as dist
        self.counts['amax_reduce']+=2
        maxima=torch.stack([x[start:start+length].float().abs().amax() for start,length in zip(STARTS,REAL)])
        self.counts['all_reduce']+=1
        self.collectives.append(dict(kind='all_reduce_max',label='destination_chunk_full_feature_amax',
            dtype=str(maxima.dtype),shape=[2],logical_peer_send_bytes=8,logical_peer_receive_bytes=8,
            self_send_bytes=0,real_rows=list(REAL),includes_model_padding_rows=53,excludes_transport_padding_rows=64))
        self.budget();dist.all_reduce(maxima,op=dist.ReduceOp.MAX)
        return (maxima.clamp_min(1e-12)*(1./2688.)).clamp_min(1e-12)

    def pack(self,x,global_scale=None):
        import torch
        import triton
        from h3_nvfp4_fastpack import pack_legacy_h3,_encode_groups
        from h3_native_nvfp4 import PackedNVFP4
        self.counts['pack']+=1
        x=x.contiguous()
        if global_scale is None:
            packed=pack_legacy_h3(x)
            self.flags.append(packed.domain_flags)
            return PackedNVFP4(packed.codes,None,packed.global_scale,packed.scales,tuple(x.shape),'original_H3')
        m,k=x.shape;rp=triton.cdiv(m,128)*128;cp=triton.cdiv(k//16,4)*4
        codes=torch.empty((m,k//2),dtype=torch.uint8,device=x.device)
        scales=torch.empty((rp*cp,),dtype=torch.float8_e4m3fn,device=x.device)
        flags=torch.zeros(2,dtype=torch.int32,device=x.device)
        _encode_groups[(triton.cdiv(rp*cp,128),)](x,global_scale,codes,scales,flags,m,k,cp,rp,128,
            num_warps=4,enable_fp_fusion=False)
        self.flags.append(flags)
        return PackedNVFP4(codes,None,global_scale,scales,tuple(x.shape),'original_H3_fixed_destination_global')

    def main(self,packet,model):
        self.counts['native_gemm']+=1
        return model.main_from_packet(packet,mode='native',include_bias=False)

    def down(self,x,model):
        import torch.nn.functional as F
        self.counts['lr_down']+=1
        return F.linear(x,model.lr_a)

    def up(self,down):
        import torch.nn.functional as F
        self.counts['lr_up']+=1
        return 1.0*F.linear(down,self.full.lr_b)

    def project_full(self,o):
        self.counts['smooth']+=1
        x=o/self.full.smooth
        packet=self.pack(x)
        down=self.down(x,self.full);branch=self.up(down)
        main=self.main(packet,self.full)
        return dict(output=main+branch,down=down,activation_globals=packet.global_scale)

    def execute(self,o,arm):
        import torch
        import torch.distributed as dist
        from h3_native_nvfp4 import PackedNVFP4
        self.counts['boundary']+=1
        if arm=='bf16_return':
            parts=[dict(o=o[start:start+length]) for start,length in zip(STARTS,REAL)]
            received=self.a2a(parts,arm)
            restored=torch.cat([p['o'] for p in received],dim=1).contiguous()
            del parts,received
            return self.project_full(restored)
        if arm=='fp8_return':
            parts=[]
            for start,length in zip(STARTS,REAL):
                self.counts['fp8_quant']+=1
                x=o[start:start+length].float()
                scale=(x.abs().amax().clamp_min(1e-12)/448.).reshape(1)
                codes=(x/scale).clamp(-448.,448.).to(torch.float8_e4m3fn).view(torch.uint8)
                parts.append(dict(codes=codes,scale=scale))
            del x,codes,scale
            received=self.a2a(parts,arm);restored=[]
            del parts
            for p in received:
                self.counts['fp8_dequant']+=1
                restored.append((p['codes'].view(torch.float8_e4m3fn).float()*p['scale']).bfloat16())
            full=torch.cat(restored,dim=1).contiguous()
            del received,restored,p
            return self.project_full(full)
        assert arm in ('row_parallel','fp4_side')
        self.counts['smooth']+=1
        x=o/self.half.smooth
        globals=self.destination_globals(x)
        if arm=='row_parallel':
            # Natural BF16 row parallelism: one coalesced RS; up only once at owner.
            send=torch.zeros((2,SLOT,DOUT+R),dtype=torch.bfloat16,device=x.device)
            for destination,(start,length) in enumerate(zip(STARTS,REAL)):
                part=x[start:start+length].contiguous()
                packet=self.pack(part,globals[destination:destination+1])
                main=self.main(packet,self.half);down=self.down(part,self.half)
                send[destination,:length,:DOUT]=main
                send[destination,:length,DOUT:]=down
            del part,packet,main,down,x
            received=torch.empty((SLOT,DOUT+R),dtype=torch.bfloat16,device=o.device)
            self.counts['reduce_scatter']+=1
            peer_bytes=SLOT*(DOUT+R)*2
            self.collectives.append(dict(kind='reduce_scatter_sum',label='BF16_partial_main_and_down32',
                dtype='torch.bfloat16',input_shape=list(send.shape),output_shape=list(received.shape),
                logical_peer_send_bytes=peer_bytes,logical_peer_receive_bytes=peer_bytes,
                extra_slot_zero_rows=[SLOT-REAL[0],SLOT-REAL[1]],
                slots=2,slot_rows=SLOT,main_features=DOUT,down_features=R,
                byte_scope='World2 RS one peer chunk per rank; logical application payload, not measured NCCL protocol traffic.'))
            self.budget();dist.reduce_scatter_tensor(received,send.reshape(2*SLOT,DOUT+R),op=dist.ReduceOp.SUM)
            main=received[:REAL[self.rank],:DOUT].contiguous()
            down=received[:REAL[self.rank],DOUT:].contiguous()
            del send,received
            return dict(output=main+self.up(down),down=down,activation_globals=globals)
        parts=[]
        for destination,(start,length) in enumerate(zip(STARTS,REAL)):
            part=x[start:start+length].contiguous()
            packet=self.pack(part,globals[destination:destination+1])
            down=self.down(part,self.half)
            parts.append(dict(codes=packet.packed,sf=packet.swizzled_scales,down=down))
        received=self.a2a(parts,arm)
        codes,sf=join_activation(received,self.rank)
        down=(received[0]['down'].float()+received[1]['down'].float()).bfloat16()
        packet=PackedNVFP4(codes,None,globals[self.rank:self.rank+1],sf,
            (REAL[self.rank],DIN),'transported_original_H3_input')
        del parts,received,part,x
        branch=self.up(down);main=self.main(packet,self.full)
        return dict(output=main+branch,down=down,activation_globals=globals)


def binding(args,report,read_input):
    import torch
    import h3_native_nvfp4 as native
    import h3_nvfp4_fastpack as fast
    import wan_native_nvfp4 as packet
    report.update(sources={k:common.record(p) for k,p in dict(runner=__file__,native=native.__file__,
        packer=fast.__file__,packet=packet.__file__,plan=args.plan).items()},
        input=common.record(args.input),weight=common.record(WEIGHT),
        geometry=dict(real_total=REAL_TOTAL,main_valid=N,heads=H,head_dim=D,input_features=DIN,output_features=DOUT,
            rank=R,token_starts=list(STARTS),real_token_lengths=list(REAL),valid_token_lengths=list(VALID),
            scale_physical_rows=list(PHYSICAL),RS_slot_rows=SLOT,model_padding_rows=53,
            row_padding_policy='Original 53 model-padding outputs are real and enter amax/projection. Only encoder SF/RS padding is artificial.'))
    payload=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert payload['shape']==[DOUT,DIN] or tuple(payload['shape'])==(DOUT,DIN)
    assert payload['tensors']['bias'] is None
    if read_input:
        value=torch.load(args.input,map_location='cpu',weights_only=True,mmap=True)
        assert torch.is_tensor(value) and value.shape==(REAL_TOTAL,H,D) and value.dtype==torch.bfloat16
        report['input_tensor']=dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(value.isfinite().all()))
        assert report['input_tensor']['finite']
    return payload


def cpu_check(report,payload):
    import torch
    from wan_native_nvfp4 import swizzle_scales,unswizzle_scales
    # Independent logical SF reconstruction checks the physical K-half concatenation.
    sf=(torch.arange(128*448).reshape(128,448)%7+1).to(torch.float8_e4m3fn)
    halves=[dict(sf=swizzle_scales(sf[:,j*224:(j+1)*224].contiguous()),
        codes=torch.full((128,HALF//2),j,dtype=torch.uint8)) for j in range(2)]
    joined=torch.cat([p['sf'].view(1,56,512) for p in halves],dim=1).reshape(-1)
    assert torch.equal(joined.view(torch.uint8),swizzle_scales(sf).view(torch.uint8))
    for rank in range(2):
        half=half_weight(payload,rank)
        fullsf=unswizzle_scales(payload['tensors']['weight_scales_swizzled'],DOUT,DIN//16)
        halvesf=unswizzle_scales(half.weight_scales_swizzled,DOUT,HALF//16)
        assert torch.equal(halvesf.view(torch.uint8),fullsf[:,rank*224:(rank+1)*224].contiguous().view(torch.uint8))
    # Tiny byte codec, unequal peer sizes and padding slots, no synthetic GEMM oracle.
    parts=[dict(o=torch.arange(m*4,dtype=torch.bfloat16).reshape(m,4)) for m in (3,5)]
    wire,sizes=encode(parts);offset=0
    for part,size,m in zip(parts,sizes,(3,5)):
        restored=decode(wire[offset:offset+size],[('o',(m,4),torch.bfloat16)])
        assert torch.equal(restored['o'],part['o']);offset+=size
    assert STARTS[1]+REAL[1]==REAL_TOTAL and sum(VALID)==N and REAL[1]-VALID[1]==53
    assert [SLOT-m for m in REAL]==[128,64]
    report['cpu_check']=dict(status='complete',scale_column_tile_rejoin=True,original_weight_half_scales=True,
        unequal_byte_payload=True,real_model_padding_included=True,native_GEMM_not_executed=True)


def run(args,report,payload,budget):
    import torch
    import torch.distributed as dist
    from h3_native_nvfp4 import NativeH3Linear
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert rank==local and int(os.environ['WORLD_SIZE'])==2 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    torch.cuda.set_device(local);assert torch.cuda.get_device_capability()==(12,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(local).total_memory))
    dist.init_process_group('nccl',timeout=timedelta(seconds=max(1,args.deadline_unix-time.time())),device_id=torch.device('cuda',local))
    report.update(rank=rank,world_size=2,environment=dict(torch=torch.__version__,cuda=torch.version.cuda,
        nccl=list(torch.cuda.nccl.version()),device=torch.cuda.get_device_name(),tf32=False,
        p2p_access_to_other_rank=bool(torch.cuda.can_device_access_peer(local,1-local))))
    raw=torch.load(args.input,map_location='cpu',weights_only=True,mmap=True)
    assert raw.shape==(REAL_TOTAL,H,D) and raw.dtype==torch.bfloat16
    o=raw[:,rank*(H//2):(rank+1)*(H//2)].reshape(REAL_TOTAL,HALF).contiguous().cuda()
    full=NativeH3Linear.from_export(payload,device='cuda')
    half=half_weight(payload,rank).cuda()
    del raw,payload
    engine=Execution(rank,full,half,budget)
    report['actual_counts']=engine.counts;report['control_barrier_calls']=0
    directory=args.data/f'rank{rank}';assert not directory.exists();directory.mkdir(parents=True)
    entries={arm:dict(arm=arm,warmups=[],repeats=[]) for arm in ARMS};report['benchmarks']=list(entries.values())
    def boundary():
        budget();dist.barrier();report['control_barrier_calls']+=1;torch.cuda.synchronize()
    def memory():
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    boundary();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    report['resident_baseline']=memory()
    def invoke(arm,index,order,position,measured):
        boundary();torch.cuda.reset_peak_memory_stats()
        before=memory();old=dict(engine.counts);engine.collectives=[];engine.flags=[]
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        started=time.perf_counter();begin.record();product=engine.execute(o,arm)
        end.record();torch.cuda.synchronize();wall=(time.perf_counter()-started)*1000
        after=memory();budget();assert after['peak_allocated']<=60*2**30
        flags=torch.stack(engine.flags).cpu()
        assert not bool(flags[:,0].ne(0).any()),'Original H3 packer reports invalid input/scale'
        receipt=dict(round=index,order=list(order),position=position,wall_ms=wall,cuda_ms=begin.elapsed_time(end),
            before=before,after=after,incremental_peak_allocated=after['peak_allocated']-before['allocated'],
            incremental_peak_reserved=after['peak_reserved']-before['reserved'],
            actual_counts={k:engine.counts[k]-old[k] for k in old},collectives=engine.collectives,
            pack_flag_checks=dict(calls=len(engine.flags),invalid_calls=0,
                calls_with_nonzero_input_zero_sf=int(flags[:,1].ne(0).sum())))
        entries[arm]['repeats' if measured else 'warmups'].append(receipt)
        if measured and index==9:
            output=product['output'].cpu();state={k:product[k].cpu() for k in ('down','activation_globals')}
            assert output.shape==(REAL[rank],DOUT) and output.dtype==torch.bfloat16 and bool(output.isfinite().all())
            assert all(bool(v.isfinite().all()) for v in state.values())
            path=directory/f'{arm}_output.pt';torch.save(output,path)
            state_path=directory/f'{arm}_state.pt';torch.save(state,state_path)
            entries[arm]['output']=dict(artifact=common.record(path),tensor=common.trecord(output))
            entries[arm]['state']=dict(artifact=common.record(state_path),tensors={k:common.trecord(v) for k,v in state.items()})
        del product
        common.save(args.output,report)
    for warm in range(3):
        for position,arm in enumerate(ARMS):invoke(arm,warm,ARMS,position,False)
    report['after_warmup_baseline']=memory()
    for repeat in range(10):
        shift=repeat%4;order=ARMS[shift:]+ARMS[:shift]
        for position,arm in enumerate(order):invoke(arm,repeat,order,position,True)
    assert engine.counts['boundary']==52 and engine.counts['native_gemm']==65
    assert engine.counts['all_to_all']==39 and engine.counts['all_reduce']==26 and engine.counts['reduce_scatter']==13
    assert engine.counts['lr_down']==78 and engine.counts['lr_up']==52 and engine.counts['pack']==78
    for entry in entries.values():
        entry.update(status='complete',median_wall_ms=statistics.median(r['wall_ms'] for r in entry['repeats']),
            median_cuda_ms=statistics.median(r['cuda_ms'] for r in entry['repeats']))
    boundary();report.update(status='complete',actual_dit_calls=0,cuda_initialized=True)
    del o,full,half,engine;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    report['released_allocated_bytes']=torch.cuda.memory_allocated();dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--output',type=Path);parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--plan',type=Path,default=PLAN)
    args=parser.parse_args();rank=int(os.environ.get('RANK','0'))
    args.output=args.output or RD/('check.json' if args.phase=='check' else f'run_rank{rank}.json')
    assert not args.output.exists(),'Preserve previous result or failure'
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=900
    else:assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E039 original deadline')
    report=dict(experiment='E039',phase=args.phase,status='running',rank=rank,world_size=2,arms=list(ARMS),
        expected_boundaries_per_rank=52,expected_native_gemms_per_rank=65,actual_dit_calls=0,
        timing_scope='Resident head-owned BF16 O through all smoothing/statistics/packs/wire copies/collectives/native projection to real token-owned projected BF16 output. JIT included in first warmup and900s deadline; control barriers, disk loading, flag readback and saving outside per-invocation timing.',
        byte_scope='Actual A2A application bytes; SUM/MAX collectives logical peer bytes separately, not NCCL protocol traffic.',
        numerical_contracts=dict(bf16_return='Original native SVD projection after BF16 A2A.',
            fp8_return='Per-source/destination message E4M3 with FP32 amax/448 scale, decode to BF16 before original smooth/projection.',
            row_parallel='Shared destination activation globals; BF16 partial main and BF16 down; one BF16 SUM RS; one BF16 up/add at owner.',
            fp4_side='Shared destination globals and actual packed input; BF16 partial down; float32 sum then BF16 down; original full main/up/add at owner.'),
        scientific_scope='One actual H3 output/projection; no quality or byte-equality gate, no DiT. Compare main-valid and real model-padding output separately.')
    try:
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        payload=binding(args,report,args.phase=='check')
        if args.phase=='check':
            cpu_check(report,payload);assert not torch.cuda.is_initialized()
            report.update(status='complete',cuda_initialized=False)
        else:
            with torch.inference_mode():run(args,report,payload,budget)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(rank=rank,status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
