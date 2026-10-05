#!/usr/bin/env python3
"""Byte-level fast-packer verification against actual DeepCompressor CUDA RTN."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party/deepcompressor'))
from deepcompressor.data.dtype import QuantDataType
from deepcompressor.quantizer.config.base import QuantizerConfig
from deepcompressor.quantizer.processor import Quantizer
from wan_native_nvfp4 import pack_activation_legacy, unswizzle_scales, PackedNVFP4
from wan_nvfp4_fastpack import pack_legacy_wan, pack_activation_fast, validate_flags, validate_quantizer_contract


def make_quantizer():
    config = QuantizerConfig(dtype=QuantDataType.from_str('sfp4_e2m1_all'),
        group_shapes=((-1,-1), (1,16)),
        scale_dtypes=(torch.float32, QuantDataType.from_str('sfp8_e4m3_nan')))
    return Quantizer(config=config, channels_dim=-1, default_dtype=None, develop_dtype=torch.float32)


def byte_diff(a,b):
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    q = make_quantizer()
    validate_quantizer_contract(q)
    report = dict(status='running', torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), started=time.time(), cases=[],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('wan_nvfp4_fastpack.py')]})
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    cases=[]
    for m,k in [(257,32),(512,1536),(31200,1536),(31200,8960)]:
        cases.append((f'random_{m}_{k}',lambda m=m,k=k:torch.randn((m,k),device='cuda',dtype=torch.bfloat16)))
    vals = [.25,.75,1.25,1.75,2.5,3.5,5.,-.25,-.75,-1.25,-1.75,-2.5,-3.5,-5.,0.,-0.]
    # Each row has exact scale1 groups; a sentinel row sets g=1.
    x=torch.tensor(vals,device='cpu',dtype=torch.bfloat16).repeat(16,2)
    x[:,15]=6; x[:,31]=6; x[-1]=2688
    cases.append(('e2m1_midpoints_signed_zero',lambda x=x:x.cuda()))
    # E4M3 representables including subnormals; group amaxes land on midpoints.
    fp8=torch.arange(0,127,dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoint=(fp8[:-1]+fp8[1:])/2
    y=(midpoint[:,None]*6).repeat(1,32).bfloat16();y[-1]=2688
    cases.append(('e4m3_midpoints_subnormal',lambda y=y:y.cuda()))
    z=torch.zeros((129,64),dtype=torch.bfloat16);z[-1]=1;z[0,0]=2**-24
    cases.append(('zero_groups_underflow',lambda z=z:z.cuda()))
    if args.inputs:
        payload=torch.load(args.inputs,map_location='cpu',weights_only=False)
        for name, value in payload.get('inputs',payload).items():
            if isinstance(value,dict):value=value.get('x_s',value.get('input'))
            if torch.is_tensor(value):cases.append((f'real_{name}',lambda value=value:value.cuda()))
    try:
        for name, factory in cases:
            x=factory().contiguous()
            before=time.time(); ref=pack_activation_legacy(x, quantizer=q)
            got=pack_legacy_wan(x); torch.cuda.synchronize()
            row=dict(name=name,shape=list(x.shape),global_scale=float(ref.global_scale),
                domain_flags=got.domain_flags.tolist(),
                byte_mismatches=dict(codes=byte_diff(got.codes,ref.packed),
                    global_scale=byte_diff(got.global_scale,ref.global_scale),
                    swizzled_scales=byte_diff(got.scales,ref.swizzled_scales)),
                seconds=time.time()-before)
            print(json.dumps(row),flush=True)
            report['cases'].append(row);save()
            validate_flags([got.domain_flags])
            if any(row['byte_mismatches'].values()):
                sf=unswizzle_scales(got.scales,ref.packed.shape[0],ref.packed.shape[1]//8).float()
                row['debug_global']=[float(got.global_scale),float(ref.global_scale)]
                row['debug_sf_first']=[sf.flatten()[:20].tolist(),ref.scales.float().flatten()[:20].tolist()]
                mismatch=(got.codes!=ref.packed).nonzero()[:10]
                row['debug_codes']=[dict(position=pos.tolist(),got=int(got.codes[tuple(pos)]),ref=int(ref.packed[tuple(pos)])) for pos in mismatch]
                save();raise AssertionError(f'packing mismatch in {name}')
            # Only timing packing, not a layer/model speedup. Includes all allocations.
            for _ in range(3):pack_legacy_wan(x)
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();start.record()
            for _ in range(10):pack_legacy_wan(x)
            end.record();end.synchronize()
            row['packing_ms_cuda']=start.elapsed_time(end)/10
            row['packing_ms_wall']=(time.perf_counter()-wall)*100
            save();del ref,got,x
        zero = torch.zeros((128,32),device='cuda',dtype=torch.bfloat16)
        zero_reference = q.quantize(zero).data
        zero_packet = pack_activation_fast(zero)
        assert torch.equal(zero_packet.decode(), zero_reference)
        report['all_zero'] = 'canonical g1 packed zero equals actual legacy QDQ; bytes intentionally not compared'
        adversarial = torch.ones((128,32),device='cuda',dtype=torch.bfloat16)
        adversarial[0] = 1e20
        adversarial_packet = pack_legacy_wan(adversarial)
        report['unrepresentable_scale_flags'] = adversarial_packet.domain_flags.tolist()
        assert report['unrepresentable_scale_flags'][1] == 1
        try:
            pack_activation_fast(adversarial)
        except ValueError:
            report['unrepresentable_scale_rejected'] = True
        else:
            raise AssertionError('unrepresentable zero SF was not rejected')
        report['nonfinite_rejected'] = {}
        for label, value in [('nan',float('nan')),('inf',float('inf')),('neg_inf',-float('inf'))]:
            bad = torch.ones((129,64), device='cuda', dtype=torch.bfloat16)
            bad[7,17] = value
            try:
                pack_activation_fast(bad)
            except ValueError:
                report['nonfinite_rejected'][label] = True
            else:
                raise AssertionError(f'{label} input was not rejected')
        report.update(status='complete' ,seconds=time.time()-report['started']);save()
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/research/E007_fastpack_parity.json')
    p.add_argument('--inputs',type=Path)
    run(p.parse_args())
