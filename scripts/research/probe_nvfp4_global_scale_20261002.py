"""CPU-only algebra/scale-range controls, not evidence of real model batch failure."""
import argparse
import json
from pathlib import Path
import torch


def quantize(x, global_scale):
    groups = x.reshape(x.shape[0], -1, 16)
    sf = (groups.abs().amax(-1) / (6 * global_scale)).to(torch.float8_e4m3fn).to(torch.float64)
    scale = sf * global_scale
    z = groups / scale.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(-1)
    levels = torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float64)
    distance = (z.abs().unsqueeze(-1)-levels).abs()
    nearest = distance.min(-1,keepdim=True).values
    # Among equal distance candidates, the E2M1 even-mantissa code wins.
    candidates = distance == nearest
    even = torch.arange(8) % 2 == 0
    has_even = (candidates & even).any(-1,keepdim=True)
    selected = candidates & (~has_even | even)
    code = selected.to(torch.int64).argmax(-1)
    q = levels[code] * z.sign() * scale.unsqueeze(-1)
    return q.reshape_as(x), sf, code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    torch.manual_seed(20261002)
    x=torch.randn(8,256,dtype=torch.float64)
    g=x.abs().max()/(6*448)
    q,sf,code=quantize(x,g)
    result={'scope':'synthetic CPU quantizer algebra; no model inputs, no latency, no native multi-request integration','controls':[]}
    for ratio in [1.,1.5,2.,16.,1024.,1e6]:
        shifted,ss,sc=quantize(x,g*ratio)
        uniform,_,_=quantize(x*ratio,g*ratio)
        result['controls'].append({'ratio':ratio,
            'co_batch_global_shift_vs_original_nmse':float((shifted-q).square().sum()/q.square().sum()),
            'shifted_vs_original_input_nmse':float((shifted-x).square().sum()/x.square().sum()),
            'code_changed_fraction':float((sc!=code).double().mean()),
            'e4m3_subnormal_nonzero_fraction':float(((ss>0)&(ss<2**-6)).double().mean()),
            'e4m3_zero_fraction':float((ss==0).double().mean()),
            'uniform_rescale_covariance_nmse':float((uniform/ratio-q).square().sum()/q.square().sum())})
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':main()
