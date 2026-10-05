#!/usr/bin/env python3
"""E040: one actual packed main per token segment, three LR information sources."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import probe_h3_query_mean_k4 as common
import bench_h3_output_projection_parallel as prior

ROOT=prior.ROOT
RD=ROOT/'results/research/E040'
DATA=Path('/data1/models/svdquant-wjq/research/20261003/E040')
PLAN=ROOT/'research_state/06_experiments/E040_projection_side_information_plan.md'
ARMS=('original_down','side_down','decoded_down')


def tensor_record(value):
    import torch
    value=value.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),finite=bool(value.float().isfinite().all()),
        sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def binding(report):
    import torch
    import h3_native_nvfp4 as native
    import h3_nvfp4_fastpack as fast
    import wan_native_nvfp4 as packet
    report['sources']={name:common.record(path) for name,path in dict(runner=__file__,plan=PLAN,
        native=native.__file__,packer=fast.__file__,packet=packet.__file__,e039=prior.__file__).items()}
    refs=[json.loads((prior.RD/f'run_rank{rank}.json').read_text()) for rank in range(2)]
    assert all(r['status']=='complete' and r['actual_dit_calls']==0 for r in refs)
    assert refs[0]['input']==refs[1]['input'] and refs[0]['weight']==refs[1]['weight']
    raw=common.load(refs[0]['input']);payload=common.load(refs[0]['weight'])
    assert raw.shape==(22592,56,128) and raw.dtype==torch.bfloat16
    t=payload['tensors']
    assert tuple(payload['shape'])==(5376,7168) and t['bias'] is None
    assert t['lr_a'].shape==(32,7168) and t['lr_b'].shape==(5376,32)
    report.update(input=refs[0]['input'],weight=refs[0]['weight'],cases=[])
    states=[]
    for rank,ref in enumerate(refs):
        rows={r['arm']:r for r in ref['benchmarks']}
        selected={arm:rows[name] for arm,name in [('original_down','bf16_return'),('side_down','fp4_side')]}
        state={arm:common.load(row['state']['artifact']) for arm,row in selected.items()}
        for arm,value in state.items():
            assert value['down'].shape==(prior.REAL[rank],32) and value['down'].dtype==torch.bfloat16
            assert tensor_record(value['down'])==selected[arm]['state']['tensors']['down']
        states.append(state)
        report['cases'].append(dict(rank=rank,start=prior.STARTS[rank],real_rows=prior.REAL[rank],
            valid_rows=prior.VALID[rank],status='prepared',e039=dict(report=common.record(prior.RD/f'run_rank{rank}.json'),
                states={arm:r['state'] for arm,r in selected.items()},
                outputs={arm:r['output'] for arm,r in selected.items()})))
    return raw,payload,states


def run(args,report,raw,payload,states,budget):
    import torch
    import torch.nn.functional as F
    from h3_native_nvfp4 import NativeH3Linear,PackedNVFP4
    from h3_nvfp4_fastpack import pack_legacy_h3
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    checked=json.loads((RD/'check.json').read_text())
    assert checked['status']=='complete' and checked['cuda_initialized'] is False
    assert checked['sources']==report['sources'] and checked['input']==report['input'] and checked['weight']==report['weight']
    report['cpu_check']=common.record(RD/'check.json')
    budget();torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(12,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(0).total_memory))
    report['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),tf32=False)
    model=NativeH3Linear.from_export(payload,device='cuda');model.eval()
    args.data.mkdir(parents=True,exist_ok=False)
    counts=report['actual_counts']
    with torch.inference_mode():
        for rank,row in enumerate(report['cases']):
            budget();started=time.monotonic()
            print(f'E040 destination {rank}, real rows {row["real_rows"]}',flush=True)
            start,length=row['start'],row['real_rows']
            x=raw[start:start+length].reshape(length,7168).contiguous().cuda()/model.smooth
            packed=pack_legacy_h3(x);counts['pack']+=1
            flags=packed.domain_flags.cpu()
            assert not bool(flags[0]),'Original H3 encoder invalid/nonfinite domain'
            packet=PackedNVFP4(packed.codes,None,packed.global_scale,packed.scales,tuple(x.shape),'original_H3_E039')
            del x,packed
            main=model.main_from_packet(packet,mode='native',include_bias=False);counts['native_gemm']+=1
            decoded=packet.decode(dtype=torch.bfloat16);counts['packet_decode']+=1
            new_down=F.linear(decoded,model.lr_a);counts['lr_down']+=1
            del decoded
            downs={arm:states[rank][arm]['down'].cuda() for arm in ARMS[:2]}
            downs['decoded_down']=new_down
            directory=args.data/f'chunk{rank}';directory.mkdir()
            packet_cpu=dict(packed=packet.packed.cpu(),swizzled_scales=packet.swizzled_scales.cpu(),
                global_scale=packet.global_scale.cpu(),original_shape=list(packet.original_shape),recipe=packet.recipe)
            path=directory/'packet.pt';torch.save(packet_cpu,path)
            row['packet']=dict(artifact=common.record(path),original_shape=packet_cpu['original_shape'],recipe=packet.recipe,
                tensors={k:tensor_record(v) for k,v in packet_cpu.items() if torch.is_tensor(v)})
            path=directory/'main.pt';main_cpu=main.cpu();assert bool(main_cpu.isfinite().all());torch.save(main_cpu,path)
            row['main']=dict(artifact=common.record(path),tensor=tensor_record(main_cpu))
            row['outputs']={}
            for arm,down in downs.items():
                budget();branch=1.0*F.linear(down,model.lr_b);counts['lr_up']+=1
                output=main+branch
                saved=output.cpu();assert saved.shape==(length,5376) and bool(saved.isfinite().all())
                path=directory/f'{arm}_output.pt';torch.save(saved,path)
                row['outputs'][arm]=dict(artifact=common.record(path),tensor=tensor_record(saved))
                del branch,output,saved
            saved_downs={arm:down.cpu() for arm,down in downs.items()}
            assert all(bool(v.isfinite().all()) for v in saved_downs.values())
            path=directory/'downs.pt';torch.save(saved_downs,path)
            row['downs']=dict(artifact=common.record(path),tensors={k:tensor_record(v) for k,v in saved_downs.items()})
            g=packet_cpu['global_scale'];old=states[rank]
            row.update(status='complete',pack_flags=dict(invalid=0,nonzero_group_zero_sf=int(flags[1])),
                global_drift=dict(original=float((g-old['original_down']['activation_globals']).abs().max()),
                    side=float((g-old['side_down']['activation_globals'][rank:rank+1]).abs().max())),
                seconds_including_io=time.monotonic()-started)
            common.save(args.output,report)
            del packet,main,main_cpu,packet_cpu,downs,new_down,down,saved_downs
            gc.collect();torch.cuda.empty_cache();budget()
    assert counts==dict(pack=2,native_gemm=2,packet_decode=2,lr_down=2,lr_up=6,communication=0,attention=0,dit=0)
    report.update(status='complete',cuda_initialized=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--output',type=Path);parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args();args.output=args.output or RD/(args.phase+'.json')
    assert not args.output.exists(),'Preserve any previous run or failure'
    report=dict(experiment='E040',phase=args.phase,status='running',arms=list(ARMS),
        actual_counts=dict(pack=0,native_gemm=0,packet_decode=0,lr_down=0,lr_up=0,communication=0,attention=0,dit=0),
        scope='Same actual packed main; only LR down information source changes. No performance/quality claim or byte-equality gate.')
    started=time.time()
    def budget():
        if args.deadline_unix and time.time()>=args.deadline_unix:raise TimeoutError('E040 original deadline')
    try:
        import torch
        torch.set_num_threads(4)
        if args.phase=='check':assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        else:assert args.deadline_unix and 0<args.deadline_unix-started<=300
        raw,payload,states=binding(report)
        if args.phase=='check':
            assert not torch.cuda.is_initialized()
            report.update(status='complete',cuda_initialized=False,
                check_scope='Actual CPU input/state shape and provenance; no native GEMM or new numerical gate.')
        else:run(args,report,raw,payload,states,budget)
    except BaseException:
        report.update(status='failed_preserved',error=traceback.format_exc());raise
    finally:
        report['seconds']=time.time()-started
        common.save(args.output,report)
        print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
