#!/usr/bin/env python3
"""E040 CPU-only same-main / three-down information-source comparison."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import summarize_h3_output_projection_parallel as common

ROOT=common.ROOT;RD=ROOT/'results/research/E040'
ARMS=('original_down','side_down','decoded_down')
REAL=common.REAL;VALID=common.VALID


def finite_tensor(value,ref,shape,dtype):
    common.require(list(value.shape)==list(shape) and value.dtype==dtype and bool(value.isfinite().all()),'Actual tensor contract')
    common.require(ref['shape']==list(shape) and ref['dtype']==str(dtype),'Saved tensor shape/dtype')
    if 'sha256' in ref:common.require(common.tensor_sha(value)==ref['sha256'],'Saved tensor provenance')
    if 'finite' in ref:common.require(ref['finite'],'Saved tensor finite')


def segmented(values,references):
    pairs=list(zip(values,references))
    return dict(valid=common.metrics([(a[:VALID[i]],b[:VALID[i]]) for i,(a,b) in enumerate(pairs)]),
        real_model_padding=common.metrics([(pairs[1][0][VALID[1]:],pairs[1][1][VALID[1]:])]),
        all_real=common.metrics(pairs))


def reduce(args):
    run_path=args.report_dir/'run.json';run=json.loads(run_path.read_text())
    common.require(run['status']=='complete','Wait for formal E040 completion')
    expected=dict(pack=2,native_gemm=2,packet_decode=2,lr_down=2,lr_up=6,communication=0,attention=0,dit=0)
    common.require(run['actual_counts']==expected,'Fixed two-packet allocation')
    common.require(common.record(run['sources']['runner']['file'])==run['sources']['runner'],'Executed producer source binding')
    old_path=args.e039_dir/'independent_summary.json';previous=json.loads(old_path.read_text())
    common.require(previous['status']=='complete','Completed E039 independent source')
    common.require(run['input']==previous['input_reference'] and run['weight']==previous['weight_reference'],'Same E039 input/weight records')
    cases=run['cases'];common.require(len(cases)==2 and [c['rank'] for c in cases]==[0,1],'Two original destination chunks')
    outputs={a:[] for a in ARMS};downs={a:[] for a in ARMS};old_outputs={a:[] for a in ('bf16_return','fp4_side')}
    provenance=[];global_drift=[]
    for rank,c in enumerate(cases):
        common.require(c['status']=='complete' and c['pack_flags']['invalid']==0,'Completed finite packet chunk')
        common.require(c['start']==(0,11264)[rank] and c['real_rows']==REAL[rank] and c['valid_rows']==VALID[rank],'True valid/model-pad support')
        p=c['packet'];packet=common.load(p['artifact'])
        common.require(packet['original_shape']==[REAL[rank],7168] and p['original_shape']==packet['original_shape'],'Unpadded original GEMM M')
        common.require(packet['recipe']==p['recipe'],'Packet recipe record')
        codes,sf,g=packet['packed'],packet['swizzled_scales'],packet['global_scale']
        common.require(codes.shape==(REAL[rank],3584) and codes.dtype==torch.uint8,'Native E2M1 packed shape')
        common.require(sf.numel()==common.PHYSICAL[rank]*448 and sf.dtype==torch.float8_e4m3fn,'Native swizzled SF extent')
        common.require(g.numel()==1 and g.dtype==torch.float32 and bool(g.isfinite().all()) and bool((g>0).all()),'Native global scalar')
        for name in ('packed','swizzled_scales','global_scale'):
            tr=p['tensors'][name]
            common.require(tr['shape']==list(packet[name].shape) and tr['dtype']==str(packet[name].dtype)
                and common.tensor_sha(packet[name])==tr['sha256'],'Actual common packet provenance')
        main=common.load(c['main']['artifact']);finite_tensor(main,c['main']['tensor'],(REAL[rank],5376),torch.bfloat16)
        ds=common.load(c['downs']['artifact']);common.require(set(ds)==set(ARMS),'Three fixed down sources')
        for arm in ARMS:
            finite_tensor(ds[arm],c['downs']['tensors'][arm],(REAL[rank],32),torch.bfloat16)
            out=common.load(c['outputs'][arm]['artifact'])
            finite_tensor(out,c['outputs'][arm]['tensor'],(REAL[rank],5376),torch.bfloat16)
            outputs[arm].append(out);downs[arm].append(ds[arm])
        old_states={}
        for old_arm,new_arm in [('bf16_return','original_down'),('fp4_side','side_down')]:
            old=previous['arms'][old_arm]
            common.require(c['e039']['states'][new_arm]==old['state_parts'][rank]
                and c['e039']['outputs'][new_arm]==old['output_parts'][rank],'Exact source artifact references')
            st=common.load(old['state_parts'][rank]['artifact']);old_states[old_arm]=st
            # This is immutable copied input identity, not an inter-method accuracy threshold.
            common.require(common.tensor_sha(st['down'])==common.tensor_sha(ds[new_arm]),'Saved E039 down input was changed')
            original=common.load(old['output_parts'][rank]['artifact'])
            finite_tensor(original,old['output_parts'][rank]['tensor'],(REAL[rank],5376),torch.bfloat16)
            old_outputs[old_arm].append(original)
        og=old_states['bf16_return']['activation_globals'].reshape(-1)[0]
        sg=old_states['fp4_side']['activation_globals'].reshape(-1)[rank]
        global_drift.append(dict(rank=rank,current=float(g.reshape(-1)[0]),e039_original=float(og),e039_side=float(sg),
            delta_original=float(g.reshape(-1)[0])-float(og),delta_side=float(g.reshape(-1)[0])-float(sg)))
        provenance.append(dict(rank=rank,packet=p,main=c['main'],downs=c['downs'],outputs=c['outputs'],
            e039=c['e039'],same_main_policy='One saved main reused by all three outputs in frozen producer; no CPU alternative GEMM introduced.'))
        del packet,main,ds,old_states
    result=dict(experiment='E040',status='running',cpu_only=True,producer_report=common.record(run_path),
        e039_summary=common.record(old_path),actual_counts=run['actual_counts'],producer_source=run['sources']['runner'],cases=provenance,activation_global_drift=global_drift,
        output_error_vs_original_down={a:segmented(outputs[a],outputs['original_down']) for a in ARMS},
        down_error_vs_original_down={a:segmented(downs[a],downs['original_down']) for a in ARMS},
        e039_replay_drift=dict(original_down=segmented(outputs['original_down'],old_outputs['bf16_return']),
                              side_down=segmented(outputs['side_down'],old_outputs['fp4_side'])),
        output_comparison_decoded_vs_side=segmented(outputs['decoded_down'],outputs['side_down']),
        source=common.record(__file__),numerical_helper=common.record(common.__file__),
        limitations=['Two chunks of one real layer/state, not independent statistical samples or quality evidence.',
            'Original and side down inputs are copied E039 states; the only new down consumes the independent decode of the same main packet.',
            'Shared main/packet is established by actual artifacts and frozen producer reuse. This reducer does not substitute a CPU GEMM to recreate BF16 GPU epilogues.',
            'Native output replay errors are reported without a bitwise scientific gate. Nonzero down difference need not imply harmful final-output or video behavior.',
            'No new communication/timing experiment; a lower local NMSE does not prove side information is required for generation quality or establish novelty.'])
    result['decoded_over_side_output_nmse']={s:(result['output_error_vs_original_down']['decoded_down'][s]['nmse']/result['output_error_vs_original_down']['side_down'][s]['nmse']
        if result['output_error_vs_original_down']['side_down'][s]['nmse'] else None) for s in ('valid','real_model_padding','all_real')}
    result.update(status='complete',cuda_initialized=torch.cuda.is_initialized())
    common.require(not result['cuda_initialized'],'CPU-only reduction')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',type=Path,default=RD)
    p.add_argument('--e039-dir',type=Path,default=ROOT/'results/research/E039');p.add_argument('--output',type=Path)
    args=p.parse_args();args.output=args.output or args.report_dir/'independent_summary.json'
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA externally');common.require(not args.output.exists(),'Preserve prior summary/failure')
    global torch
    import torch
    torch.set_num_threads(6);common.torch=torch;start=time.monotonic()
    result=dict(experiment='E040',status='failed_stop',source=common.record(__file__))
    try:result=reduce(args)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:
        result['seconds']=time.monotonic()-start;common.save(args.output,result)
        print(json.dumps(dict(status=result['status'],output=str(args.output),sha256=common.record(args.output)['sha256'])))


if __name__=='__main__':main()
