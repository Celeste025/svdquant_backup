#!/usr/bin/env python3
"""E093: distribution of native NVFP4 Q/DQ error for one real H3 FC2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import FP4_LEVELS, KEY, WEIGHT, dequantize, load_input


EXPECTED = Path('results/research/E090/h3_fc2_p11cal_p1eval_smoothquant_nvfp4_diagnostic.json')
REGIONS = {
    'p1': {'text': (0, 658), 'audio': (658, 1072),
           'video': (1072, 22384), 'padding': (22384, 22400)},
    'p11': {'text': (0, 738), 'audio': (738, 1152),
            'video': (1152, 22464), 'padding': (22464, 22464)},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p1', type=Path, required=True)
    p.add_argument('--p11', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=7)
    return p.parse_args()


def value(x):
    return float(x.item())


def percentiles(x):
    if not x.numel():
        return None
    q = torch.quantile(x.float(), torch.tensor([0.5, 0.9, 0.99], device=x.device))
    return dict(p50=value(q[0]), p90=value(q[1]), p99=value(q[2]))


def top_share(x, percent):
    n = max(1, math.ceil(x.numel() * percent / 100))
    return dict(count=n, fraction=value(torch.topk(x, n).values.sum() / x.sum().clamp_min(1e-30)))


def top_indices(error, signal, n=8):
    v, idx = torch.topk(error, min(n, error.numel()))
    total_e, total_s = error.sum().clamp_min(1e-30), signal.sum().clamp_min(1e-30)
    return [dict(index=int(i), error_share=value(e / total_e),
                 signal_share=value(signal[i] / total_s),
                 local_nmse=value(e / signal[i].clamp_min(1e-30)))
            for e, i in zip(v, idx, strict=True)]


def quantized_error(x, levels, nvfp4_quantize, sf_layout):
    amax = value(x.abs().amax().float())
    inverse = torch.tensor([2688.0 / amax], dtype=torch.float32, device=x.device)
    q, sf = nvfp4_quantize(x, inverse, sfLayout=sf_layout, backend='cuda')
    row_e = torch.empty(x.shape[0], device=x.device, dtype=torch.float64)
    row_s = torch.empty_like(row_e)
    col_e = torch.zeros(x.shape[1], device=x.device, dtype=torch.float64)
    col_s = torch.zeros_like(col_e)
    for start in range(0, x.shape[0], 128):
        stop = min(start + 128, x.shape[0])
        decoded = dequantize(q[start:stop], sf[start:stop],
                             inverse.reciprocal(), levels)
        orig = x[start:stop].float()
        error2 = (decoded - orig).square()
        signal2 = orig.square()
        row_e[start:stop] = error2.sum(dim=1, dtype=torch.float64)
        row_s[start:stop] = signal2.sum(dim=1, dtype=torch.float64)
        col_e += error2.sum(dim=0, dtype=torch.float64)
        col_s += signal2.sum(dim=0, dtype=torch.float64)
        del decoded, orig, error2, signal2
    del q, sf
    return dict(amax=amax, row_error=row_e, row_signal=row_s,
                channel_error=col_e, channel_signal=col_s)


def summarize(data, regions=None):
    re, rs = data['row_error'], data['row_signal']
    ce, cs = data['channel_error'], data['channel_signal']
    total_e, total_s = re.sum(), rs.sum()
    if not torch.allclose(total_e, ce.sum(), rtol=1e-10, atol=0):
        raise RuntimeError('Row and channel error sums disagree')
    if not torch.allclose(total_s, cs.sum(), rtol=1e-10, atol=0):
        raise RuntimeError('Row and channel energy sums disagree')
    group_e, group_s = ce.reshape(-1, 16).sum(dim=1), cs.reshape(-1, 16).sum(dim=1)
    output = dict(shape=[int(re.numel()), int(ce.numel())], global_amax=data['amax'],
                  global_nmse=value(total_e / total_s),
                  total_error_energy=value(total_e), total_signal_energy=value(total_s),
                  rows=dict(top_1pct_error_share=top_share(re, 1.0),
                            top_1pct_signal_share=top_share(rs, 1.0),
                            top_0_1pct_error_share=top_share(re, 0.1),
                            local_nmse_percentiles=percentiles(re / rs.clamp_min(1e-30)),
                            top_error=top_indices(re, rs)),
                  input_channels=dict(top_1pct_error_share=top_share(ce, 1.0),
                                      top_1pct_signal_share=top_share(cs, 1.0),
                                      local_nmse_percentiles=percentiles(ce / cs.clamp_min(1e-30)),
                                      top_error=top_indices(ce, cs)),
                  k16_groups=dict(top_1pct_error_share=top_share(group_e, 1.0),
                                  top_1pct_signal_share=top_share(group_s, 1.0),
                                  local_nmse_percentiles=percentiles(group_e / group_s.clamp_min(1e-30)),
                                  top_error=top_indices(group_e, group_s)))
    if regions is not None:
        output['modality'] = {}
        for name, (lo, hi) in regions.items():
            e, s = re[lo:hi], rs[lo:hi]
            output['modality'][name] = dict(rows=hi-lo,
                error_energy_share=value(e.sum() / total_e),
                signal_energy_share=value(s.sum() / total_s),
                nmse=value(e.sum() / s.sum().clamp_min(1e-30)) if hi > lo else None,
                row_local_nmse_percentiles=percentiles(e / s.clamp_min(1e-30)))
        for row in output['rows']['top_error']:
            row['modality'] = next((name for name, (lo, hi) in regions.items()
                                    if lo <= row['index'] < hi), 'unknown')
    return output


def main():
    a = parse_args()
    if a.p1.resolve() == a.p11.resolve():
        raise ValueError('Different real calls required')
    from flashinfer import SfLayout, nvfp4_quantize
    expected = json.loads(EXPECTED.read_text())['rows'][0]
    p1_cpu, p1_call = load_input(a.p1)
    p11_cpu, p11_call = load_input(a.p11)
    if p1_call == p11_call or p1_cpu.shape[0] != 22400 or p11_cpu.shape[0] != 22464:
        raise ValueError('Unexpected H3 call identity or shape')
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        weight_cpu = f.get_tensor(KEY)
    if tuple(weight_cpu.shape) != (5376, 14336):
        raise ValueError('Unexpected H3 FC2 weight shape')
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    levels = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=device)
    report = dict(format='h3-nvfp4-operand-error-distribution-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='one H3 block-25 FC2 BF16 weight and p1/p11 step-0 input; plain native NVFP4 Q/DQ',
                  quantization='FlashInfer native NVFP4 E2M1, E4M3 K16 blocks, current-input FP32 global scale; weight global scale from weight',
                  p1=str(a.p1), p11=str(a.p11), p1_call=p1_call, p11_call=p11_call,
                  weight=str(WEIGHT), weight_key=KEY, gpu=a.gpu, operands={})
    with torch.inference_mode():
        for name, cpu, regions, gate in (
            ('p1_activation', p1_cpu, REGIONS['p1'], expected['calls']['evaluation']['dynamic_global_activation_qdq_nmse']),
            ('p11_activation', p11_cpu, REGIONS['p11'], expected['calls']['calibration']['activation_qdq_nmse']),
            ('weight', weight_cpu, None, expected['weight_qdq_nmse']),
        ):
            x = cpu.to(device)
            data = quantized_error(x, levels, nvfp4_quantize, SfLayout.layout_linear)
            result = summarize(data, regions)
            delta = abs(result['global_nmse'] - gate)
            if delta > 2e-7:
                raise RuntimeError(f'{name} E090 operand NMSE gate failed: {delta}')
            result['e090_global_nmse_gate_delta'] = delta
            report['operands'][name] = result
            print(f'{name}: NMSE={result["global_nmse"]:.8f}, '
                  f'row top1% share={result["rows"]["top_1pct_error_share"]["fraction"]:.3f}, '
                  f'channel top1% share={result["input_channels"]["top_1pct_error_share"]["fraction"]:.3f}',
                  flush=True)
            del x, data
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'wrote {a.output}', flush=True)


if __name__ == '__main__':
    main()
