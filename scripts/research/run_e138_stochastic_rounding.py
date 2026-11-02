#!/usr/bin/env python3
"""Activation-only stochastic E2M1 rounding, matched SVDQuant H3 rollouts."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/research')]
import minimax_h3_svdquant_common as common

# Retain the exact existing scale computation and BF16 writeback.  Only the
# E2M1 code selection changes, with saturation outside the representable range.
source = inspect.getsource(common.nvfp4_qdq).replace('def nvfp4_qdq(', 'def stochastic_qdq(')
old = 'idx = (z.unsqueeze(-1) - levels).abs().argmin(dim=-1)'
new = '''hi = torch.searchsorted(levels, z.contiguous()).clamp(1, levels.numel()-1)
        lo = hi - 1
        probability = ((z-levels[lo])/(levels[hi]-levels[lo])).clamp(0, 1)
        idx = torch.where(torch.rand(z.shape, device=z.device, generator=SR_GENERATOR) < probability, hi, lo)'''
assert source.count(old) == 1
SR_GENERATOR = None
namespace = dict(vars(common), SR_GENERATOR=None)
exec(source.replace(old, new), namespace)
stochastic_qdq = namespace['stochastic_qdq']


def validate():
    global SR_GENERATOR
    SR_GENERATOR = torch.Generator(device='cuda').manual_seed(138)
    namespace['SR_GENERATOR'] = SR_GENERATOR
    # Repeated identical groups: the empirical mean must approach the original
    # normalized value inside the codebook.  FP8 scale and BF16 cast remain.
    x = torch.linspace(-5.5, 5.5, 16, device='cuda').repeat(65536, 1)
    y = stochastic_qdq(x, element_size=1024)
    nearest = common.nvfp4_qdq(x, element_size=1024)
    bias = float((y.mean(0)-x[0]).abs().max())
    assert bias < 0.02, bias
    assert torch.isfinite(y).all()
    result = {'max_empirical_mean_bias': bias, 'repetitions': 65536,
              'nearest_mse': float((nearest-x).square().mean()),
              'stochastic_mse': float((y-x).square().mean()),
              'scope': 'activation E2M1 only; FP8 local scales retain nearest rounding'}
    path = ROOT / 'results/research/E138/rounding_validation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


@torch.inference_mode()
def local_check():
    payload = Path('/data1/models/svdquant-wjq/research/20261015/E102_h3_fc_scale/p1_step0_b0_fc1_input_weight.pt')
    p = torch.load(payload, map_location='cpu', weights_only=False)
    x = p['input'].reshape(-1, p['input'].shape[-1]).cuda()
    w, s, a, b = [p[k].cuda() for k in ['weight', 'smooth', 'a', 'b']]
    xs = x / s
    wq = common.nvfp4_qdq(w*s-b@a)
    # Retain the global scale of the entire real input, but evaluate a bounded
    # set of output rows to keep this diagnostic smaller than generation.
    rows = min(2048, x.shape[0])
    reference = torch.nn.functional.linear(x[:rows], w).float()
    correction = torch.nn.functional.linear(torch.nn.functional.linear(xs[:rows], a), b)
    results = []
    for seed in [None, 1380001, 1380002, 1380003]:
        if seed is None:
            q = common.nvfp4_qdq(xs)
        else:
            namespace['SR_GENERATOR'] = torch.Generator(device='cuda').manual_seed(seed)
            q = stochastic_qdq(xs)
        output = (torch.nn.functional.linear(q[:rows], wq) + correction).float()
        error = output-reference
        centered = error-error.mean(0, keepdim=True)
        cosine = torch.nn.functional.cosine_similarity(centered[:-1], centered[1:], dim=-1).mean()
        results.append({'seed': seed, 'rounding': 'nearest' if seed is None else 'stochastic',
                        'output_nmse': float(error.square().sum()/reference.square().sum()),
                        'activation_nmse': float((q.float()-xs.float()).square().sum()/xs.float().square().sum()),
                        'centered_adjacent_row_error_cosine': float(cosine)})
        del q, output, error, centered
    result = {'source': str(payload), 'layer': p['layer'], 'output_rows': rows,
              'input_shape': list(x.shape), 'scope': 'first contiguous rows; not a spatial or temporal neighbor assertion',
              'results': results}
    target = ROOT/'results/research/E138/local_fc_check.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=['race', 'clap0'])
    ap.add_argument('--arm', choices=['nearest', 'activation_sr'])
    ap.add_argument('--phase', choices=['validate', 'local', 'denoise', 'decode'], required=True)
    args = ap.parse_args()
    if args.phase == 'validate':
        return validate()
    if args.phase == 'local':
        return local_check()
    import run_e129_h3_svdquant_attention as e129
    import run_e121_h3_split_generation as e121
    base = e121.base
    base.REPORT = ROOT / 'results/research/E138'
    base.DATA = Path('/data1/models/svdquant-wjq/research/20261031/E138_stochastic_rounding')
    base.PLAN = Path(__file__)
    class Router(e121.Router):
        def __init__(self, dit, arm):
            super().__init__(dit, 'bf16')
    base.Router = Router
    def install(dit):
        # Weight quantization completes with the original nearest-rounding
        # function before changing the dynamic activation hook's global lookup.
        description = e129.install_linears(dit)
        if args.arm == 'activation_sr':
            namespace['SR_GENERATOR'] = torch.Generator(device='cuda').manual_seed(1380001)
            common.nvfp4_qdq = stochastic_qdq
        description.update(activation_rounding=args.arm, weight_rounding='nearest',
                           rounding_seed=1380001)
        return description
    base.checked_linears = install
    torch.set_num_threads(6)
    if args.phase == 'denoise':
        base.denoise(args.case, args.arm)
    else:
        base.decode(args.case, args.arm)


if __name__ == '__main__':
    main()
