#!/usr/bin/env python3
"""E037 CPU-only two-rank global/coarse comparison, preserving distinct contracts."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import summarize_h3_sequence_parallel_correction as previous

cpu=previous.cpu
RD=previous.ROOT/'results/research/E037'
ARMS=('fp4_table16','fp4_global')
N,H,D=22539,56,128
LENGTHS,VALID=previous.LENGTHS,previous.VALID
save,distribution=previous.save,previous.distribution


def expected_counts(arm):
    return dict(q_center_reduce_ops=8 if arm=='fp4_table16' else 0,k_mean_ops=0,k_sum_ops=1,
        q_sum_ops=int(arm=='fp4_global'),q_pack=1,k_pack=1,v_pack=1,correction_gemm=1,attention=1,
        all_to_all=3 if arm=='fp4_table16' else 2,all_reduce=1)


def global_collectives(pair):
    totals=[]
    for rank,receipt in enumerate(pair):
        rows=receipt['collectives'];peer_rows=pair[1-rank]['collectives']
        assert [r['label'] for r in rows]==['padded_Q_and_valid_K_sums','forward_fp4_global','inverse_BF16_output']
        result=dict(a2a_remote_send_bytes=0,a2a_remote_receive_bytes=0,a2a_self_send_bytes=0,
                    allreduce_logical_peer_send_bytes=0,allreduce_logical_peer_receive_bytes=0,by_label={})
        reduction=rows[0]
        assert reduction['kind']=='all_reduce_sum' and reduction['dtype']=='torch.float32'
        assert reduction['shape']==[2,H,1,D] and reduction['input_numel']==reduction['output_numel']==2*H*D
        assert reduction['element_size']==4 and reduction['self_send_bytes']==0
        size=2*H*D*4
        assert reduction['logical_peer_send_bytes']==reduction['logical_peer_receive_bytes']==size
        assert reduction['fields']==dict(padded_Q_sum=dict(bytes=size//2,denominator=22656),
                                          valid_K_sum=dict(bytes=size//2,denominator=N))
        result['allreduce_logical_peer_send_bytes']=result['allreduce_logical_peer_receive_bytes']=size
        result['by_label'][reduction['label']]=dict(logical_peer_send_bytes=size,logical_peer_receive_bytes=size)
        for i,c in enumerate(rows[1:],start=1):
            peer=peer_rows[i];assert c['kind']==peer['kind']=='all_to_all_single' and c['label']==peer['label']
            assert c['dtype']=='torch.uint8' and c['element_size']==1
            send,recv=c['send_splits'],c['recv_splits']
            assert len(send)==len(recv)==2 and sum(send)==c['input_numel'] and sum(recv)==c['output_numel']
            assert send[rank]==recv[rank]==c['self_send_bytes']==c['self_receive_bytes']
            assert send[1-rank]==peer['recv_splits'][rank]==c['remote_send_bytes']
            assert recv[1-rank]==peer['send_splits'][rank]==c['remote_receive_bytes']
            expected=(28*LENGTHS[rank]*(3*(D//2+D//16)+4) if i==1
                      else 28*LENGTHS[1-rank]*D*2)
            assert send[1-rank]==expected
            if i==1:
                table=c['fields']['table'];assert table['shape']==[H,1,LENGTHS[rank]]
                assert table['dtype']=='torch.float32' and table['bytes']==H*LENGTHS[rank]*4
            result['a2a_remote_send_bytes']+=c['remote_send_bytes']
            result['a2a_remote_receive_bytes']+=c['remote_receive_bytes']
            result['a2a_self_send_bytes']+=c['self_send_bytes']
            result['by_label'][c['label']]=dict(remote_send_bytes=c['remote_send_bytes'],
                remote_receive_bytes=c['remote_receive_bytes'],self_send_bytes=c['self_send_bytes'])
        totals.append(result)
    combined={k:sum(r[k] for r in totals) for k in totals[0] if k!='by_label'}
    assert combined['a2a_remote_send_bytes']==combined['a2a_remote_receive_bytes']
    combined['a2a_remote_send_mib']=combined['a2a_remote_send_bytes']/2**20
    combined['allreduce_logical_peer_send_mib']=combined['allreduce_logical_peer_send_bytes']/2**20
    combined['inverse_output_remote_send_bytes']=sum(r['by_label']['inverse_BF16_output']['remote_send_bytes'] for r in totals)
    combined['forward_and_center_a2a_remote_send_bytes']=combined['a2a_remote_send_bytes']-combined['inverse_output_remote_send_bytes']
    return dict(per_rank=totals,combined=combined)


def receipts(entries,arm):
    ledger=None
    for measured,label,count in ((False,'warmups',3),(True,'repeats',10)):
        assert all(len(e[label])==count for e in entries)
        for index in range(count):
            pair=[e[label][index] for e in entries]
            order=ARMS if not measured or index%2==0 else ARMS[::-1]
            for row in pair:
                assert row['round']==index and row['order']==list(order) and row['position']==order.index(arm)
                assert row['actual_counts']==expected_counts(arm)
                assert all(math.isfinite(row[k]) and row[k]>0 for k in ('wall_ms','cuda_ms'))
                for name in ('allocated','reserved'):
                    assert row['after']['peak_'+name]>=max(row['before'][name],row['after'][name])
                    assert row['incremental_peak_'+name]==row['after']['peak_'+name]-row['before'][name]
                assert row['after']['peak_allocated']<=60*2**30
            now=previous.check_collectives(pair,arm) if arm=='fp4_table16' else global_collectives(pair)
            if ledger is None:ledger=now
            else:assert ledger==now
    walls=[[r['wall_ms'] for r in e['repeats']] for e in entries]
    events=[[r['cuda_ms'] for r in e['repeats']] for e in entries]
    memory=[]
    for rank,entry in enumerate(entries):
        assert entry['status']=='complete'
        assert math.isclose(statistics.median(walls[rank]),entry['median_wall_ms'],rel_tol=1e-12)
        assert math.isclose(statistics.median(events[rank]),entry['median_cuda_ms'],rel_tol=1e-12)
        mem=dict(rank=rank)
        for name in ('allocated','reserved'):
            mem['resident_before_'+name]=distribution([r['before'][name] for r in entry['repeats']])
            mem['peak_'+name]=distribution([r['after']['peak_'+name] for r in entry['repeats']])
            mem['incremental_peak_'+name]=distribution([r['incremental_peak_'+name] for r in entry['repeats']])
        memory.append(mem)
    return dict(synchronized_wall_pair_max_ms=distribution([max(a,b) for a,b in zip(*walls)]),
        per_rank_wall_ms=[distribution(v) for v in walls],per_rank_cuda_ms=[distribution(v) for v in events],
        memory_per_rank=memory,maximum_single_device_peak_allocated=max(m['peak_allocated']['max'] for m in memory),
        maximum_single_device_peak_reserved=max(m['peak_reserved']['max'] for m in memory),
        communication_per_invocation=ledger)


def reduce(args,report):
    import torch
    torch.set_num_threads(6);cpu.torch=torch;cpu.DEADLINE=time.monotonic()+180
    paths=[args.report_dir/f'run_rank{rank}.json' for rank in (0,1)]
    ranks=[json.loads(p.read_text()) for p in paths]
    expected={k:13*sum(expected_counts(a)[k] for a in ARMS) for k in expected_counts(ARMS[0])}
    for rank,source in enumerate(ranks):
        assert source['status']=='complete' and source['rank']==source['local_rank']==rank and source['world_size']==2
        assert source['arms']==list(ARMS) and source['actual_dit_calls']==0 and source['control_barrier_calls']==28
        assert source['actual_counts']==expected and source['actual_counts']['attention']==26
        assert source['environment']['visible_devices']=='0,1' and source['environment']['tf32'] is False
    for key in ('sources','reference_report','input','references','geometry','private_source_hash','e036_reports'):
        assert ranks[0][key]==ranks[1][key],key
    ref=ranks[0]['references'];geo=ranks[0]['geometry']
    assert geo['valid_token_lengths']==list(VALID) and geo['padded_token_lengths']==list(LENGTHS)
    assert geo['global_contract']['padded_q_denominator']==22656 and geo['global_contract']['valid_k_denominator']==N
    assert geo['global_contract']['table_rows']==1 and geo['head_ranges']==[[0,28],[28,56]]
    assert cpu.record(ranks[0]['sources']['runner']['file'])==ranks[0]['sources']['runner']
    assert cpu.record(ranks[0]['reference_report']['file'])==ranks[0]['reference_report']
    references={name:cpu.load(ref[name]['artifact']) for name in ('bf16','coarse16','global')}
    for name,value in references.items():
        assert value.shape==(N,H,D) and value.dtype==torch.bfloat16
        if 'sha256' in ref[name].get('tensor',{}):assert cpu.tensor_sha(value)==ref[name]['tensor']['sha256']
    report.update(rank_reports=[cpu.record(p) for p in paths],runner=ranks[0]['sources']['runner'],
        references=ref,reference_report=ranks[0]['reference_report'],geometry=geo,
        actual_native_attention_calls=sum(s['actual_counts']['attention'] for s in ranks),actual_dit_calls=0,
        rank_receipts=[{k:s[k] for k in ('rank','environment','actual_counts','control_barrier_calls',
            'resident_baseline','after_warmup_baseline','released_allocated_bytes')} for s in ranks],arms={})
    outputs,states={},{}
    for arm in ARMS:
        cpu.budget();entries=[next(e for e in s['benchmarks'] if e['arm']==arm) for s in ranks]
        result=receipts(entries,arm);pieces=[];parts=[];centers=16 if arm=='fp4_table16' else 1
        for rank,entry in enumerate(entries):
            output=cpu.load(entry['output']['artifact']);state=cpu.load(entry['state']['artifact'])
            assert output.shape==(VALID[rank],H,D) and output.dtype==torch.bfloat16 and bool(output.isfinite().all())
            assert entry['output']['tensor']['shape']==list(output.shape) and entry['output']['tensor']['finite'] is True
            for name,shape in (('centers',(28,centers,D)),('k_mean',(28,1,D))):
                value=state[name];assert value.shape==shape and value.dtype==torch.bfloat16 and bool(value.isfinite().all())
                assert cpu.tensor_sha(value)==entry['state']['tensors'][name]['sha256']
            pieces.append(output);parts.append(state)
        outputs[arm]=torch.cat(pieces,dim=0)
        states[arm]={name:torch.cat([s[name] for s in parts],dim=0) for name in ('centers','k_mean')}
        result.update(output_parts=[e['output'] for e in entries],state_parts=[e['state'] for e in entries],
            reconstructed_output=dict(shape=list(outputs[arm].shape),dtype=str(outputs[arm].dtype),sha256=cpu.tensor_sha(outputs[arm])),
            own_contract_reference='coarse16' if centers==16 else 'global')
        report['arms'][arm]=result;save(args.output,report)
    for arm in ARMS:
        refs=dict(references)
        if arm=='fp4_global':refs['current_fp4_table16']=outputs['fp4_table16']
        report['arms'][arm]['output_error']=cpu.metrics(outputs[arm],refs)
    first=report['arms']['fp4_table16'];second=report['arms']['fp4_global']
    before=first['synchronized_wall_pair_max_ms']['median'];after=second['synchronized_wall_pair_max_ms']['median']
    e0=first['output_error']['bf16'];e1=second['output_error']['bf16']
    report['paired_comparison']=dict(global_over_coarse_wall_ratio=after/before,
        global_wall_change_fraction=after/before-1,global_speedup=before/after,
        paired_round_wall_ratios=[b/a for a,b in zip(first['synchronized_wall_pair_max_ms']['values'],second['synchronized_wall_pair_max_ms']['values'])],
        global_nmse_minus_coarse=e1['pooled']['nmse']-e0['pooled']['nmse'],
        global_nmse_over_coarse=e1['pooled']['nmse']/e0['pooled']['nmse'],
        per_head=[dict(head=h,coarse_nmse=a['nmse'],global_nmse=b['nmse'],difference=b['nmse']-a['nmse'])
                  for h,(a,b) in enumerate(zip(e0['per_head'],e1['per_head']))],
        global_better_heads=sum(b['nmse']<a['nmse'] for a,b in zip(e0['per_head'],e1['per_head'])),
        global_worse_heads=sum(b['nmse']>a['nmse'] for a,b in zip(e0['per_head'],e1['per_head'])),
        global_equal_heads=sum(b['nmse']==a['nmse'] for a,b in zip(e0['per_head'],e1['per_head'])))
    report['k_mean_global_vs_coarse']=cpu.metrics(states['fp4_global']['k_mean'].transpose(0,1),
        dict(coarse=states['fp4_table16']['k_mean'].transpose(0,1)))['coarse']
    # Same recipe in the previous run: retain drift separately, never substitute historical timing.
    old=[]
    for file_ref in ranks[0]['e036_reports']:
        assert cpu.record(file_ref['file'])==file_ref
        data=json.loads(Path(file_ref['file']).read_text());assert data['status']=='complete'
        old.append(next(e for e in data['benchmarks'] if e['arm']=='fp4_table16'))
    old_times=[max(old[0]['repeats'][i]['wall_ms'],old[1]['repeats'][i]['wall_ms']) for i in range(10)]
    report['historical_t16_timing_context']=dict(reports=ranks[0]['e036_reports'],pair_max_wall_ms=distribution(old_times),
        current_over_historical_median=before/statistics.median(old_times),scope='Diagnostic run-to-run context only; primary comparison uses current paired arms.')
    cpu.budget();assert not torch.cuda.is_initialized()
    report.update(status='complete',cuda_initialized=False)
    for arm,row in report['arms'].items():
        print(json.dumps(dict(arm=arm,wall_ms=row['synchronized_wall_pair_max_ms']['median'],
            peer_send_mib=row['communication_per_invocation']['combined']['a2a_remote_send_mib'],
            nmse_bf16=row['output_error']['bf16']['pooled']['nmse'],
            nmse_own_reference=row['output_error'][row['own_contract_reference']]['pooled']['nmse'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=RD);p.add_argument('--output',type=Path,default=RD/'independent_summary.json')
    args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not args.output.exists(),'Preserve existing summary/failure'
    started=time.monotonic()
    report=dict(experiment='E037',status='running',source=cpu.record(__file__),
        helpers=dict(e036_reducer=cpu.record(previous.__file__),fp64_metrics=cpu.record(cpu.__file__)),
        scope='CPU-only saved outputs/states/receipts; no original QKV/model read or GPU.',
        methods='Each round: max of two rank synchronized wall times, then median/range. Separate global/coarse historical-reference drift and common-BF16 FP64 per-head error. Sum only remote A2A sends; AllReduce logical payload separate.',
        limitations=['Distinct center/consumer contracts; no requirement or claim of equal outputs or video quality.',
            'One observed layer, sample, shape and two-card topology; not whole-model deployment speedup.',
            'Peak memory is per rank, not sum of non-simultaneous rank peaks. Shared allocator reserved memory is preserved.',
            'Application payload is not measured NCCL protocol traffic; CUDA stage times are not added.'])
    try:reduce(args,report)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.monotonic()-started;save(args.output,report)


if __name__=='__main__':main()
