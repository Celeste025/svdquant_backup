#!/usr/bin/env python3
"""E036 independent CPU reduction: paired-rank timing, payloads and outputs.

No distributed import/collective, CUDA, source QKV, model, or kernel execution.
Per-rank wall arrays are matched by round before taking the maximum.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import summarize_h3_transferred_centers as cpu

ROOT=Path(__file__).resolve().parents[2]
RD=ROOT/'results/research/E036'
ARMS=('bf16_a2a','mixed_k16','fp4_table16')
LENGTHS=(11264,11392)
VALID=(11264,11275)
N,H,D,K=22539,56,128,16


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def distribution(values):
    return dict(values=values,min=min(values),median=statistics.median(values),max=max(values),mean=statistics.mean(values))


def expected_counts(arm):
    return dict(q_center_reduce_ops=16 if arm=='bf16_a2a' else 8,
        k_mean_ops=int(arm!='fp4_table16'),k_sum_ops=int(arm=='fp4_table16'),
        q_pack=1,k_pack=1,v_pack=1,correction_gemm=1,attention=1,
        all_to_all=3 if arm=='fp4_table16' else 2,all_reduce=int(arm=='fp4_table16'))


def forward_remote_bytes(arm,length):
    # One destination's 28 heads. All packed scale bytes are transmitted too.
    if arm=='bf16_a2a': return 28*length*D*2*3
    if arm=='mixed_k16': return 28*length*(2*(D//2+D//16)+D*2)+28*8*D*2
    return 28*length*3*(D//2+D//16)+28*K*length*4


def check_collectives(pair,arm):
    """Verify cross-rank send/receive split conservation; sum only peer sends."""
    lists=[r['collectives'] for r in pair]
    expected_labels=([f'forward_{arm}','inverse_BF16_output'] if arm!='fp4_table16'
                     else ['centers_all_heads','valid_K_sum',f'forward_{arm}','inverse_BF16_output'])
    assert all([c['label'] for c in rows]==expected_labels for rows in lists)
    totals=[]
    for rank,rows in enumerate(lists):
        result=dict(a2a_remote_send_bytes=0,a2a_remote_receive_bytes=0,a2a_self_send_bytes=0,
                    allreduce_logical_peer_send_bytes=0,allreduce_logical_peer_receive_bytes=0,by_label={})
        for i,c in enumerate(rows):
            peer=lists[1-rank][i];assert c['kind']==peer['kind'] and c['label']==peer['label']
            if c['kind']=='all_to_all_single':
                assert c['dtype']=='torch.uint8' and c['element_size']==1
                send,recv=c['send_splits'],c['recv_splits']
                assert len(send)==len(recv)==2
                assert sum(send)==c['input_numel'] and sum(recv)==c['output_numel']
                assert send[1-rank]==peer['recv_splits'][rank] and recv[1-rank]==peer['send_splits'][rank]
                assert send[rank]==recv[rank]==c['self_send_bytes']==c['self_receive_bytes']
                assert send[1-rank]==c['remote_send_bytes'] and recv[1-rank]==c['remote_receive_bytes']
                if c['label'].startswith('forward_'): expected=forward_remote_bytes(arm,LENGTHS[rank])
                elif c['label']=='centers_all_heads': expected=H*8*D*2
                else: expected=28*LENGTHS[1-rank]*D*2
                assert c['remote_send_bytes']==expected,(arm,rank,c['label'])
                result['a2a_remote_send_bytes']+=c['remote_send_bytes']
                result['a2a_remote_receive_bytes']+=c['remote_receive_bytes']
                result['a2a_self_send_bytes']+=c['self_send_bytes']
                result['by_label'][c['label']]=dict(remote_send_bytes=c['remote_send_bytes'],
                    remote_receive_bytes=c['remote_receive_bytes'],self_send_bytes=c['self_send_bytes'])
            else:
                assert c['kind']=='all_reduce_sum' and c['dtype']=='torch.float32' and c['element_size']==4
                size=H*D*4
                assert c['input_numel']==c['output_numel']==H*D
                assert c['logical_peer_send_bytes']==c['logical_peer_receive_bytes']==size
                assert c['self_send_bytes']==0
                result['allreduce_logical_peer_send_bytes']+=size
                result['allreduce_logical_peer_receive_bytes']+=size
                result['by_label'][c['label']]=dict(logical_peer_send_bytes=size,logical_peer_receive_bytes=size)
        totals.append(result)
    keys=('a2a_remote_send_bytes','a2a_remote_receive_bytes','a2a_self_send_bytes',
          'allreduce_logical_peer_send_bytes','allreduce_logical_peer_receive_bytes')
    combined={k:sum(r[k] for r in totals) for k in keys}
    assert combined['a2a_remote_send_bytes']==combined['a2a_remote_receive_bytes']
    combined['a2a_remote_send_mib']=combined['a2a_remote_send_bytes']/2**20
    combined['allreduce_logical_peer_send_mib']=combined['allreduce_logical_peer_send_bytes']/2**20
    combined['inverse_output_remote_send_bytes']=sum(r['by_label']['inverse_BF16_output']['remote_send_bytes'] for r in totals)
    combined['forward_and_center_a2a_remote_send_bytes']=combined['a2a_remote_send_bytes']-combined['inverse_output_remote_send_bytes']
    return dict(per_rank=totals,combined=combined)


def check_receipt(receipt,arm,round_index,warmup):
    order=ARMS if warmup else ARMS[round_index%3:]+ARMS[:round_index%3]
    assert receipt['round']==round_index and receipt['order']==list(order)
    assert receipt['position']==order.index(arm) and receipt['actual_counts']==expected_counts(arm)
    assert receipt['wall_ms']>0 and receipt['cuda_ms']>0
    assert math.isfinite(receipt['wall_ms']) and math.isfinite(receipt['cuda_ms'])
    before,after=receipt['before'],receipt['after']
    for name in ('allocated','reserved'):
        assert after['peak_'+name]>=max(before[name],after[name])
        assert receipt['incremental_peak_'+name]==after['peak_'+name]-before[name]
    assert after['peak_allocated']<=60*2**30


def arm_timings(entries,arm):
    ledger=None
    for warmup,label,count in ((True,'warmups',3),(False,'repeats',10)):
        assert all(len(e[label])==count for e in entries)
        for i in range(count):
            pair=[e[label][i] for e in entries]
            for r in pair:check_receipt(r,arm,i,warmup)
            got=check_collectives(pair,arm)
            if ledger is None:ledger=got
            else:assert got==ledger,'Fixed-shape communication payload changed between repetitions'
    walls=[[e['repeats'][i]['wall_ms'] for i in range(10)] for e in entries]
    cuda=[[e['repeats'][i]['cuda_ms'] for i in range(10)] for e in entries]
    result=dict(synchronized_wall_pair_max_ms=distribution([max(walls[0][i],walls[1][i]) for i in range(10)]),
        per_rank_wall_ms=[distribution(x) for x in walls],per_rank_cuda_ms=[distribution(x) for x in cuda],
        communication_per_invocation=ledger,memory_per_rank=[])
    for rank,entry in enumerate(entries):
        assert entry['status']=='complete'
        assert math.isclose(statistics.median(walls[rank]),entry['median_wall_ms'],rel_tol=1e-12)
        assert math.isclose(statistics.median(cuda[rank]),entry['median_cuda_ms'],rel_tol=1e-12)
        samples=entry['repeats']
        row=dict(rank=rank)
        for name in ('allocated','reserved'):
            row['resident_before_'+name]=distribution([s['before'][name] for s in samples])
            row['peak_'+name]=distribution([s['after']['peak_'+name] for s in samples])
            row['incremental_peak_'+name]=distribution([s['incremental_peak_'+name] for s in samples])
        result['memory_per_rank'].append(row)
    result['maximum_single_device_peak_allocated']=max(r['peak_allocated']['max'] for r in result['memory_per_rank'])
    result['maximum_single_device_peak_reserved']=max(r['peak_reserved']['max'] for r in result['memory_per_rank'])
    return result


def reduce(args,report):
    import torch
    torch.set_num_threads(6);cpu.torch=torch;cpu.DEADLINE=time.monotonic()+180
    paths=[args.report_dir/f'run_rank{r}.json' for r in (0,1)]
    sources=[json.loads(p.read_text()) for p in paths]
    for rank,s in enumerate(sources):
        assert s['status']=='complete' and s['rank']==s['local_rank']==rank and s['world_size']==2
        assert s['arms']==list(ARMS) and s['actual_dit_calls']==0
        assert s['actual_counts']=={k:13*sum(expected_counts(a)[k] for a in ARMS) for k in expected_counts(ARMS[0])}
        assert s['actual_counts']['attention']==39 and s['control_barrier_calls']==41
        assert s['environment']['visible_devices']=='0,1' and s['environment']['tf32'] is False
    for name in ('sources','reference_report','input','references','geometry','private_source_hash'):
        assert sources[0][name]==sources[1][name],name
    geometry=sources[0]['geometry']
    assert geometry['valid_length']==N and geometry['padded_length']==22656
    assert geometry['padded_token_lengths']==list(LENGTHS) and geometry['valid_token_lengths']==list(VALID)
    assert geometry['head_ranges']==[[0,28],[28,56]] and geometry['center_boundaries']==[j*177//16 for j in range(17)]
    assert cpu.record(sources[0]['sources']['runner']['file'])==sources[0]['sources']['runner']
    assert cpu.record(sources[0]['reference_report']['file'])==sources[0]['reference_report']
    references={name:cpu.load(sources[0]['references'][name]['artifact']) for name in ('bf16','coarse16')}
    for name,value in references.items():
        assert value.shape==(N,H,D) and value.dtype==torch.bfloat16
        row=sources[0]['references'][name].get('tensor',{})
        if 'sha256' in row:assert cpu.tensor_sha(value)==row['sha256']
    previous_centers=cpu.load(sources[0]['references']['coarse_centers']['artifact'])['centers']
    assert previous_centers.shape==(H,K,D) and previous_centers.dtype==torch.bfloat16
    report.update(rank_reports=[cpu.record(p) for p in paths],runner=sources[0]['sources']['runner'],
        reference_report=sources[0]['reference_report'],references=sources[0]['references'],geometry=geometry,
        actual_native_attention_calls=sum(s['actual_counts']['attention'] for s in sources),actual_dit_calls=0,
        rank_receipts=[dict(rank=s['rank'],environment=s['environment'],actual_counts=s['actual_counts'],
            control_barrier_calls=s['control_barrier_calls'],resident_baseline=s['resident_baseline'],
            after_warmup_baseline=s['after_warmup_baseline'],released_allocated_bytes=s['released_allocated_bytes']) for s in sources],arms={})
    outputs,states={},{}
    for arm in ARMS:
        cpu.budget();entries=[next(e for e in s['benchmarks'] if e['arm']==arm) for s in sources]
        row=arm_timings(entries,arm)
        pieces=[];state_parts=[]
        for rank,entry in enumerate(entries):
            output=cpu.load(entry['output']['artifact'])
            assert output.shape==(VALID[rank],H,D) and output.dtype==torch.bfloat16 and bool(output.isfinite().all())
            assert entry['output']['tensor']['shape']==list(output.shape) and entry['output']['tensor']['finite'] is True
            state=cpu.load(entry['state']['artifact'])
            for name,shape in (('centers',(28,K,D)),('k_mean',(28,1,D))):
                value=state[name];receipt=entry['state']['tensors'][name]
                assert value.shape==shape and value.dtype==torch.bfloat16 and bool(value.isfinite().all())
                assert cpu.tensor_sha(value)==receipt['sha256']
            pieces.append(output);state_parts.append(state)
        outputs[arm]=torch.cat(pieces,dim=0)
        states[arm]={name:torch.cat([s[name] for s in state_parts],dim=0) for name in ('centers','k_mean')}
        row.update(output_parts=[e['output'] for e in entries],state_parts=[e['state'] for e in entries],
            reconstructed_output=dict(shape=list(outputs[arm].shape),dtype=str(outputs[arm].dtype),sha256=cpu.tensor_sha(outputs[arm])))
        report['arms'][arm]=row;save(args.output,report)
    base=report['arms']['bf16_a2a'];base_time=base['synchronized_wall_pair_max_ms']['median']
    base_payload=base['communication_per_invocation']['combined']['a2a_remote_send_bytes']
    for index,arm in enumerate(ARMS):
        cpu.budget();row=report['arms'][arm]
        comparisons={**references,**{other:outputs[other] for other in ARMS[:index]}}
        row['output_error']=cpu.metrics(outputs[arm],comparisons)
        center_ref={'e034_coarse16':previous_centers.transpose(0,1)}
        if arm!='bf16_a2a':center_ref['bf16_a2a']=states['bf16_a2a']['centers'].transpose(0,1)
        row['center_error']=cpu.metrics(states[arm]['centers'].transpose(0,1),center_ref)
        row['k_mean_error_vs_bf16_a2a']=cpu.metrics(states[arm]['k_mean'].transpose(0,1),
            dict(bf16_a2a=states['bf16_a2a']['k_mean'].transpose(0,1)))['bf16_a2a']
        row['k_mean_changed_elements_vs_bf16_a2a']=int((states[arm]['k_mean']!=states['bf16_a2a']['k_mean']).sum())
        median=row['synchronized_wall_pair_max_ms']['median']
        payload=row['communication_per_invocation']['combined']['a2a_remote_send_bytes']
        row['relative_to_bf16_a2a']=dict(median_wall_ratio=median/base_time,speedup=base_time/median,
            median_wall_change_fraction=median/base_time-1,a2a_remote_payload_ratio=payload/base_payload,
            paired_round_wall_ratios=[a/b for a,b in zip(row['synchronized_wall_pair_max_ms']['values'],base['synchronized_wall_pair_max_ms']['values'])])
        save(args.output,report)
        print(json.dumps(dict(arm=arm,paired_wall_median_ms=median,peer_send_mib=payload/2**20,
            nmse_bf16=row['output_error']['bf16']['pooled']['nmse'],
            nmse_coarse=row['output_error']['coarse16']['pooled']['nmse'])),flush=True)
    report['cross_arm_head_ordering']={arm:dict(
        lower_error_heads=int(sum(a['nmse']<b['nmse'] for a,b in zip(report['arms'][arm]['output_error']['bf16']['per_head'],base['output_error']['bf16']['per_head']))),
        higher_error_heads=int(sum(a['nmse']>b['nmse'] for a,b in zip(report['arms'][arm]['output_error']['bf16']['per_head'],base['output_error']['bf16']['per_head']))))
        for arm in ARMS[1:]}
    cpu.budget();assert not torch.cuda.is_initialized()
    report.update(status='complete',cuda_initialized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,default=RD)
    p.add_argument('--output',type=Path,default=RD/'independent_summary.json')
    args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not args.output.exists(),'Preserve prior summary/failure'
    started=time.monotonic()
    report=dict(experiment='E036',status='running',source=cpu.record(__file__),metrics_helper=cpu.record(cpu.__file__),
        scope='Independent CPU reduction of final token-owned outputs, small center/Kmean state and complete two-rank receipts; no original QKV/model reread.',
        timing='Each of ten matching rounds uses max(rank0 wall, rank1 wall), then median/range. CUDA event arrays remain per-rank diagnostics and are never summed with stages.',
        communication='Sum peer sends over both ranks once. A2A actual application split payload excludes self and receives; AllReduce logical tensor payload is separate. No NCCL protocol/link measurement.',
        memory='Per-rank resident/after-warmup and repeat baselines/peaks retained; report largest single-device peak, not sum of non-simultaneous rank peaks.',
        limitations=['One shape, sample and layer on the recorded two-GPU topology; no full-model or quality result.',
            'Different K reduction and table GEMM partitions may produce numeric drift; no cross-arm byte scientific gate.',
            'A2A peer bytes and AllReduce logical tensor bytes are not measured network transport traffic.',
            'Three warmups/ten repeats per arm are performance observations, not independent model-quality samples.'])
    try:reduce(args,report)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.monotonic()-started;save(args.output,report)


if __name__=='__main__':main()
