#!/usr/bin/env python3
"""Post-E018 exploratory CPU projection, not a flicker/perception measurement."""
import json
import os
from pathlib import Path
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import summarize_h3_plain_baseline as util

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E018'
INHERITED_SHA = '7338985b4a561561f8c33018a6f08b7d3b97418785b08e45cb67126f829a9950'
ARMS = ('block_0', 'block_64', 'block_128', 'global_0', 'delta_64_0')

def main():
    output = RD/'exploratory_coherence.json'
    util.require(not output.exists(), 'Refusing to overwrite exploratory result')
    started = time.monotonic(); torch.set_num_threads(6)
    files = util.Files()
    inherited = files.verify(RD/'independent_summary.json', INHERITED_SHA)
    summary = json.loads(Path(inherited['file']).read_text())
    util.require(summary['status'] == 'complete', 'Inherited audit incomplete')
    probe_path = RD/'probe_run.json'
    files.record(summary['files'][str(probe_path)])
    probe = json.loads(probe_path.read_text())
    c = (-1.)**torch.arange(2,35,dtype=torch.float64); c -= c.mean()
    cnorm = float(c.square().sum())
    report = dict(status='running',scope='Post-E018 exploratory, fixed raster-vector coherence; no GPU',
        inherited_independent_summary=inherited,inherited_verified_files=summary['verified_files'],
        source={str(Path(__file__).resolve()):util.file_sha(__file__), str(Path(util.__file__).resolve()):util.file_sha(util.__file__)},
        methods=dict(frames=[2,34],within_frame_tokens=576,heads=56,head_dim=128,threads=6,dtype='FP64 CPU',
            basis=c.tolist(),basis_square_sum=cnorm,chunk_within_frame_positions=16,
            formula='A=sum_t c_t E_t / sum_t c_t^2; projection energy=sum_t c_t^2 * ||A||^2',
            limits='No 1/33 significance threshold; no motion alignment or perceptual flicker interpretation; one state and three correlated layers'),blocks=[])
    try:
        for case in probe['cases']:
            tensors = {}
            for mode,shift,key in [('bf16',0,'reference'),('block_mean',0,'block_0'),('block_mean',64,'block_64'),
                                   ('block_mean',128,'block_128'),('global_mean',0,'global_0')]:
                row = next(r for r in case['outputs'] if r['mode']==mode and r['shift']==shift)
                path = files.record(row['output']['artifact'])['file']
                value = util.load_tensor_file(path)
                util.require(util.tensor_record(value)==row['output']['tensor'],'Read output tensor SHA mismatch')
                util.require(tuple(value.shape)==(22539,56,128) and value.dtype==torch.bfloat16,'Geometry drift')
                tensors[key] = value[1227:].reshape(37,576,56,128)[2:35]
            sums = {a:dict(total_error_energy=0.,a_energy=0.) for a in ARMS}
            dot = residual_energy = residual_max = 0.
            for start in range(0,576,16):
                sl = slice(start,start+16)
                ref = tensors['reference'][:,sl].double()
                errors = {a:tensors[a][:,sl].double()-ref for a in ARMS[:-1]}
                util.require(torch.equal(tensors['block_128'][:,sl].contiguous().view(torch.uint8),
                                         tensors['block_0'][:,sl].contiguous().view(torch.uint8)), '128 vector control differs')
                errors['delta_64_0'] = errors['block_64']-errors['block_0']
                util.require(torch.equal(errors['delta_64_0'],tensors['block_64'][:,sl].double()-tensors['block_0'][:,sl].double()),'Delta vector identity failed')
                projections = {}
                for arm,e in errors.items():
                    util.require(bool(torch.isfinite(e).all()),'Nonfinite error')
                    a = (c[:,None,None,None]*e).sum(0)/cnorm
                    projections[arm] = a
                    sums[arm]['total_error_energy'] += float(e.square().sum())
                    sums[arm]['a_energy'] += float(a.square().sum())
                dot += float((projections['block_0']*projections['block_64']).sum())
                residual = projections['delta_64_0']-(projections['block_64']-projections['block_0'])
                residual_energy += float(residual.square().sum()); residual_max=max(residual_max,float(residual.abs().max()))
            for value in sums.values():
                value['projection_energy'] = cnorm*value['a_energy']
                value['projection_fraction'] = util.ratio(value['projection_energy'],value['total_error_energy'])
                util.require(value['projection_energy'] <= value['total_error_energy']*(1+1e-12),'Projection exceeds total energy')
            identity_rhs=sums['block_64']['a_energy']+sums['block_0']['a_energy']-2*dot
            identity_error=sums['delta_64_0']['a_energy']-identity_rhs
            util.require(abs(identity_error)<=1e-10*max(sums['delta_64_0']['a_energy'],1e-30),'Projection identity failed')
            util.require(residual_energy<=1e-20*max(sums['delta_64_0']['a_energy'],1e-30),'Projection linearity failed')
            report['blocks'].append(dict(block=case['block'],arms=sums,
                A0_A64_cosine=util.ratio(dot,(sums['block_0']['a_energy']*sums['block_64']['a_energy'])**.5),
                A0_A64_dot=dot,controls=dict(block128_vector_byte_exact=True,delta_vector_identity_exact=True,
                    projection_linearity_residual_energy=residual_energy,projection_linearity_max_abs=residual_max,
                    delta_projection_energy_identity_residual=cnorm*identity_error)))
            print('E018 coherence block',case['block'],'complete',flush=True)
        util.require([b['block'] for b in report['blocks']]==[0,24,48],'Layer selection drift')
        util.require(not torch.cuda.is_initialized(),'CUDA initialized')
        report.update(status='complete',files=files.checked)
    except BaseException as exc:
        report.update(status='failed_stop',error=repr(exc)); raise
    finally:
        report.update(seconds_total=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized())
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(report['status'],report['seconds_total'],flush=True)

if __name__=='__main__': main()
