#!/usr/bin/env python3
"""E034 equal-capacity contiguous coarse centers versus fixed existing baselines.

156 full-path attention calls (three saved cases, four arms, 3 warm + 10 timed),
zero DiT. Reuses the frozen E033 consumer and packet APIs without new kernels.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback
import bench_h3_codebook_consumer as previous
import probe_h3_query_mean_k4 as common
import probe_h3_codebook_centers as codebook
import probe_h3_adaptive_centers as adaptive

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E034'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E034')
PLAN=ROOT/'research_state/06_experiments/E034_coarse_center_baseline_plan.md'
H,N,NP,D,G,K=56,22539,22656,128,177,16
ARMS=('global','fullblock','codebook','coarse16')
BOUNDARIES=tuple(j*G//K for j in range(K+1))


def coarse_ids_cpu():
    import torch
    ids=torch.empty(G,dtype=torch.int32)
    for j,(lo,hi) in enumerate(zip(BOUNDARIES[:-1],BOUNDARIES[1:])):ids[lo:hi]=j
    return ids[None,None].expand(1,H,G).contiguous()


def binding(report,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    import codebook_attention_sm120 as private
    path=ROOT/'results/research/E033/run.json'
    prior=json.loads(path.read_text())
    assert prior['status']=='complete' and prior['consumer_control']['passed']
    assert prior['validation_attention_calls']==10 and prior['benchmark_attention_calls']==468
    # The executed consumer, quantizers, construction and benchmark helpers stay frozen.
    current=dict(runner=previous.__file__,codebook=codebook.__file__,common=common.__file__,
                 adaptive=adaptive.__file__,official=official.__file__,private=private.__file__)
    for key,value in current.items():assert common.record(value)==prior['sources'][key],key
    report['sources']={name:common.record(path) for name,path in dict(
        runner=__file__,previous=previous.__file__,codebook=codebook.__file__,common=common.__file__,
        adaptive=adaptive.__file__,official=official.__file__,private=private.__file__,plan=PLAN).items()}
    report['reference_report']=common.record(path)
    report['private_source']=private.source_manifest()
    assert report['private_source']==prior['private_source']
    ids=coarse_ids_cpu()
    assert BOUNDARIES[0]==0 and BOUNDARIES[-1]==G and len(BOUNDARIES)==K+1
    assert all(a<b for a,b in zip(BOUNDARIES[:-1],BOUNDARIES[1:]))
    for j,(lo,hi) in enumerate(zip(BOUNDARIES[:-1],BOUNDARIES[1:])):
        assert torch.equal(ids[0,0,lo:hi],torch.full((hi-lo,),j,dtype=torch.int32))
    assert sum(b-a for a,b in zip(BOUNDARIES[:-1],BOUNDARIES[1:]))==G
    assert ids.shape==(1,H,G) and int(ids.min())==0 and int(ids.max())==15
    private.validate_center_ids(ids,K)
    report['coarse_geometry']=dict(boundaries=list(BOUNDARIES),ids=previous.trecord(ids),
        intervals=[dict(center=j,group_start=a,group_end=b,token_start=128*a,padded_token_end=128*b,
            valid_tokens=max(0,min(N,128*b)-128*a),mean_denominator=128*(b-a),
            padding_tokens=128*(b-a)-max(0,min(N,128*b)-128*a)) for j,(a,b) in enumerate(zip(BOUNDARIES[:-1],BOUNDARIES[1:]))],
        map_policy='Assign j over [floor(j*G/16), floor((j+1)*G/16)); not floor(g*16/G).',
        static_metadata='The fixed geometry ID map is uploaded once outside timing and resident for all arms; data-dependent centers are recomputed every call.')
    report['inputs']=[]
    for block in (0,24,48):
        budget();source=next(row for row in prior['inputs'] if row['block']==block)
        raw=common.load(source['capture'])
        for name in ('q','k','v'):assert raw[name].shape==(N,H,D) and raw[name].dtype==torch.bfloat16
        assert raw['valid_length']==N and raw['scale']==D**-.5
        refs={name:next(row['output'] for row in prior['benchmark'] if row['block']==block and row['arm']==name)
              for name in ARMS if name!='coarse16'}
        refs['bf16']=source['outputs']['bf16']
        for reference in refs.values():
            value=common.load(reference['artifact']);assert value.shape==(N,H,D) and value.dtype==torch.bfloat16
        report['inputs'].append(dict(block=block,capture=source['capture'],outputs=refs))
    return ids


def run(args,report,ids_cpu,budget):
    import torch
    import flashinfer.nvfp4_attention_sm120 as official
    import codebook_attention_sm120 as private
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and torch.cuda.get_device_capability()==(12,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(0).total_memory))
    outputs=DATA/'outputs';assert not outputs.exists();outputs.mkdir(parents=True)
    report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        visible_gpu=0,tf32=False,cuda_graph=False)
    module=official.get_nvfp4_attention_sm120_module();private.get_codebook_module();budget()
    weights=torch.full((1,G),128.,dtype=torch.float32,device='cuda');weights[:,-1]=11
    fixed_ids=ids_cpu.cuda();private.validate_center_ids(fixed_ids,K)
    report['resident_metadata_bytes']=dict(coarse_ids=fixed_ids.numel()*fixed_ids.element_size(),
        codebook_weights=weights.numel()*weights.element_size())
    fields=('preprocess','k_mean','q_global_mean','q_block_mean','q_coarse_mean','q_coarse_stack',
            'construct','q_pack','k_pack','v_pack','correction_gemm','official_attention','private_attention','attention')
    counts={key:0 for key in fields};report['actual_counts']=counts;report['benchmark']=[]

    def memory():
        return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())

    def check_memory():
        budget();assert torch.cuda.max_memory_allocated()<=60*2**30,'Allocation exceeds 60 GiB'

    def prepare_pack(dense,mode):
        q,k,v=dense
        counts['preprocess']+=1;counts['k_mean']+=1
        ids=None
        if mode=='codebook':
            qpad,kc,mu,global_mu=adaptive.means_and_key(q,k)
            counts['q_block_mean']+=1;counts['q_global_mean']+=1
            built=codebook.construct(mu,global_mu,weights);counts['construct']+=1
            centers=built['centers'];ids=built['ids'].unsqueeze(0).contiguous()
            selected=centers.gather(1,ids[0].long().unsqueeze(-1).expand(-1,-1,D))
            qcenter=(qpad.reshape(H,G,128,D)-selected.unsqueeze(2)).reshape(H,NP,D).contiguous()
        else:
            kc=torch.nn.functional.pad(k-k.mean(-2,keepdim=True),(0,0,0,NP-N))
            qpad=torch.nn.functional.pad(q,(0,0,0,NP-N))
            if mode=='global':
                centers=qpad.mean(-2,keepdim=True);counts['q_global_mean']+=1
                qcenter=(qpad-centers).contiguous()
            elif mode=='fullblock':
                centers=qpad.reshape(H,G,128,D).mean(2);counts['q_block_mean']+=1
                qcenter=(qpad.reshape(H,G,128,D)-centers.unsqueeze(2)).reshape(H,NP,D).contiguous()
            else:
                assert mode=='coarse16'
                # Direct BF16 padded-token reductions, not averages of pre-rounded block means.
                centers=torch.stack([qpad[:,128*lo:128*hi].mean(1)
                    for lo,hi in zip(BOUNDARIES[:-1],BOUNDARIES[1:])],dim=1)
                counts['q_coarse_mean']+=K;counts['q_coarse_stack']+=1
                ids=fixed_ids
                selected=centers.gather(1,ids[0].long().unsqueeze(-1).expand(-1,-1,D))
                qcenter=(qpad.reshape(H,G,128,D)-selected.unsqueeze(2)).reshape(H,NP,D).contiguous()
        vpad=torch.nn.functional.pad(v,(0,0,0,NP-N))
        q4=torch.empty((1,H,NP,D//2),device='cuda',dtype=torch.uint8)
        qs=torch.empty((1,H,NP,D//16),device='cuda',dtype=torch.float8_e4m3fn)
        k4,ks=torch.empty_like(q4),torch.empty_like(qs)
        vt=torch.empty((1,H,D,NP//2),device='cuda',dtype=torch.uint8)
        vs=torch.empty((1,H,D,NP//16),device='cuda',dtype=torch.float8_e4m3fn)
        module.scaled_fp4_quant(qcenter.unsqueeze(0),q4,qs,1);counts['q_pack']+=1
        module.scaled_fp4_quant_permute(kc.unsqueeze(0),k4,ks,1);counts['k_pack']+=1
        module.scaled_fp4_quant_trans(vpad.unsqueeze(0),vt,vs,1);counts['v_pack']+=1
        # Match E033: release dense padded/centered temporaries before the table/native stage.
        return (q4,k4,vt,qs,ks,vs),centers,kc,ids

    def execute(dense,mode):
        packets,centers,kc,ids=prepare_pack(dense,mode)
        kt=kc.transpose(-2,-1).float()  # Same single conversion/lifetime as E033.
        table=(centers.float()@kt).unsqueeze(0).contiguous()
        counts['correction_gemm']+=1
        assert report['attention_calls']<156
        report['attention_calls']+=1;counts['attention']+=1
        if ids is not None:
            counts['private_attention']+=1
            result=private.codebook_fwd(*packets,table,ids,sm_scale=D**-.5,unpadded_k_len=N)
        else:
            counts['official_attention']+=1
            result=official.nvfp4_attention_sm120_fwd(*packets,table,sm_scale=D**-.5,
                causal=False,per_block_mean=mode!='global',out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)
        # Small centers/ids are returned by this same call for final-call provenance.
        # No extra reconstruction or attention call is used to save them.
        return dict(output=result[0,:,:N].transpose(0,1).contiguous(),centers=centers,ids=ids)

    def benchmark(source,dense,mode,refs):
        check_memory();gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry=dict(block=source['block'],arm=mode,status='warming',warmup_calls=3,
            resident_baseline=memory(),repeats=[],counts_before=dict(counts))
        report['benchmark'].append(entry);common.save(args.output,report)
        for _ in range(3):
            check_memory();products=execute(dense,mode);del products
        torch.cuda.synchronize();entry.update(status='measuring',after_warmup_baseline=memory())
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for index in range(10):
            check_memory();torch.cuda.reset_peak_memory_stats()
            before,counts_before=memory(),dict(counts)
            started=time.perf_counter();begin.record()
            products=execute(dense,mode)
            end.record();end.synchronize()
            elapsed=(time.perf_counter()-started)*1000;after=memory()
            entry['repeats'].append(dict(index=index,cuda_ms=begin.elapsed_time(end),synchronized_wall_ms=elapsed,
                before=before,after=after,incremental_peak_allocated=after['peak_allocated']-before['allocated'],
                incremental_peak_reserved=after['peak_reserved']-before['reserved'],
                actual_counts={key:counts[key]-counts_before[key] for key in counts}))
            common.save(args.output,report)
            if index!=9:del products
        cpu=products['output'].cpu()
        if mode=='coarse16':
            saved=dict(centers=products['centers'].cpu(),ids=products['ids'].cpu(),
                boundaries=torch.tensor(BOUNDARIES,dtype=torch.int32))
            assert saved['centers'].shape==(H,K,D) and saved['centers'].dtype==torch.bfloat16
            assert torch.equal(saved['ids'],ids_cpu) and bool(saved['centers'].isfinite().all())
            path=outputs/f'block{source["block"]}_coarse16_centers.pt';torch.save(saved,path)
            entry['centers']=dict(artifact=common.record(path),tensors={key:previous.trecord(value) for key,value in saved.items()},
                provenance='Returned small tensors from repeat index9 of the measured execute; no extra GPU reconstruction.')
        del products
        path=outputs/f'block{source["block"]}_{mode}.pt';torch.save(cpu,path)
        entry['output']=dict(artifact=common.record(path),tensor=previous.trecord(cpu))
        assert entry['output']['tensor']['finite']
        entry['versus_bf16']=previous.numeric(cpu,refs['bf16'])
        compare=('global','fullblock','codebook') if mode=='coarse16' else (mode,)
        entry['versus_e033']={name:previous.numeric(cpu,refs[name]) for name in compare}
        entry['actual_counts']={key:counts[key]-entry['counts_before'][key] for key in counts}
        for name in ('preprocess','k_mean','q_pack','k_pack','v_pack','correction_gemm','attention'):
            assert entry['actual_counts'][name]==13,(mode,name)
        assert entry['actual_counts']['construct']==(13 if mode=='codebook' else 0)
        assert entry['actual_counts']['q_global_mean']==(13 if mode in ('global','codebook') else 0)
        assert entry['actual_counts']['q_block_mean']==(13 if mode in ('fullblock','codebook') else 0)
        assert entry['actual_counts']['q_coarse_mean']==(13*K if mode=='coarse16' else 0)
        assert entry['actual_counts']['q_coarse_stack']==(13 if mode=='coarse16' else 0)
        assert entry['actual_counts']['private_attention']==(13 if mode in ('codebook','coarse16') else 0)
        assert entry['actual_counts']['official_attention']==(13 if mode in ('global','fullblock') else 0)
        entry.update(status='complete',median_cuda_ms=statistics.median(row['cuda_ms'] for row in entry['repeats']),
            median_synchronized_wall_ms=statistics.median(row['synchronized_wall_ms'] for row in entry['repeats']),
            peak_allocated=max(row['after']['peak_allocated'] for row in entry['repeats']),
            peak_reserved=max(row['after']['peak_reserved'] for row in entry['repeats']),
            max_incremental_peak_allocated=max(row['incremental_peak_allocated'] for row in entry['repeats']),
            max_incremental_peak_reserved=max(row['incremental_peak_reserved'] for row in entry['repeats']))
        common.save(args.output,report)
        print(json.dumps(dict(block=source['block'],arm=mode,cuda_ms=entry['median_cuda_ms'],peak_gib=entry['peak_allocated']/2**30)),flush=True)

    for source in report['inputs']:
        check_memory();raw=common.load(source['capture'])
        dense=tuple(raw[name].transpose(0,1).contiguous().cuda() for name in ('q','k','v'))
        refs={name:common.load(value['artifact']) for name,value in source['outputs'].items()}
        for mode in ARMS:benchmark(source,dense,mode,refs)
        del raw,dense,refs
        gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    assert report['attention_calls']==counts['attention']==156
    assert counts['private_attention']==counts['official_attention']==78
    assert counts['construct']==39 and counts['q_coarse_mean']==39*K
    check_memory();report.update(status='complete',cuda_initialized=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--deadline-unix',type=float);parser.add_argument('--output',type=Path)
    args=parser.parse_args();args.output=args.output or RD/(args.phase+'.json');assert not args.output.exists()
    started=time.time()
    if args.phase=='run':assert args.deadline_unix and 0<args.deadline_unix-started<=900
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E034 shared deadline')
    report=dict(experiment='E034',phase=args.phase,status='running',actual_dit_calls=0,attention_calls=0,
        arms=list(ARMS),warmups=3,repeats=10,expected_attention_calls=156,
        geometry=dict(heads=H,valid_length=N,padded_length=NP,head_dim=D,query_groups=G,coarse_centers=K),
        timing_scope='Resident contiguous BF16 HND QKV plus fixed geometry IDs to valid contiguous NHD output. Include necessary BF16 reductions/pad/centering, fresh clustering only for codebook, all three packs, FP32 K conversion, table GEMM, native attention and output assembly. Exclude upload/static-ID construction/JIT/CPU hashes/files/numeric reports.',
        memory_scope='Per-arm cache clear; all arms retain the same fixed ID map and weights. Warm baseline then per-repeat resetpeak/baseline/peak/increment. Every call returns its small center/ID tensors along with output; all products are released before the next repeat. Final coarse small tensors are saved from that call without re-execution.',
        scope='Same K16 capacity contiguous-center strong baseline; no new kernel, video, parameter search, byte scientific gate or novel-algorithm claim.')
    try:
        import torch
        torch.set_num_threads(6)
        if args.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        ids=binding(report,budget)
        if args.phase=='check':
            assert not torch.cuda.is_initialized();report.update(status='complete',cuda_initialized=False)
        else:
            checked=json.loads((RD/'check.json').read_text());assert checked['status']=='complete' and not checked['cuda_initialized']
            for key in ('sources','reference_report','private_source','inputs','coarse_geometry'):assert checked[key]==report[key],key
            with torch.inference_mode():run(args,report,ids,budget)
    except BaseException:
        report.update(status='failed_stop',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started;common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
