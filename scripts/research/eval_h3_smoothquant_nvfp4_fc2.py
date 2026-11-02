#!/usr/bin/env python3
"""E090: error-only SmoothQuant-style channel smoothing + native NVFP4 W4A4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import (FP4_LEVELS, INPUT_KEY, KEY, WEIGHT,
                                      dequantize, load_input)


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_P1 = Path('results/research/E089/h3_fc2_p1_s00_row_scales.json')
EXPECTED_P11 = Path('results/research/E089/h3_fc2_p11_s00_row_scales.json')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=7)
    p.add_argument('--max-rows', type=int, default=0,
                   help='smoke subset only; skips full-shape E089 baseline gate')
    p.add_argument('--activation-stat', choices=('absmax', 'abs-p99'), default='absmax',
                   help='per-input-channel calibration statistic used only for smoothing')
    return p.parse_args()


def squared_error_ratio(a, b):
    a32, b32 = a.float(), b.float()
    return float(((a32 - b32).square().sum() / b32.square().sum()).item())


def qdq_error(x, inverse, levels, nvfp4_quantize, sf_layout):
    q, sf = nvfp4_quantize(x, inverse, sfLayout=sf_layout, backend='cuda')
    numerator = torch.zeros((), device=x.device, dtype=torch.float64)
    denominator = torch.zeros((), device=x.device, dtype=torch.float64)
    for start in range(0, x.shape[0], 128):
        stop = min(start + 128, x.shape[0])
        restored = dequantize(q[start:stop], sf[start:stop], inverse.reciprocal(), levels)
        original = x[start:stop].float()
        numerator += (restored - original).square().sum().double()
        denominator += original.square().sum().double()
    del q, sf
    return float((numerator / denominator).item())


def expected_baselines():
    p1 = json.loads(EXPECTED_P1.read_text())
    p11 = json.loads(EXPECTED_P11.read_text())
    return {
        'evaluation': p1['rows'][0]['output_nmse'],
        'evaluation_dynamic_global': next(r['output_nmse'] for r in p1['rows']
                                          if r['activation_scale'] == 'eval_tensor' and
                                          r['weight_scale'] == 'tensor'),
        'calibration': next(r['output_nmse'] for r in p11['rows']
                            if r['activation_scale'] == 'eval_tensor' and
                            r['weight_scale'] == 'tensor'),
    }


def activation_channel_stat(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == 'absmax':
        return x.abs().amax(dim=0).float().clamp_min(1e-5)
    # Exact nearest-rank empirical p99, calculated in channel chunks so no
    # additional full-size FP32 copy or sorted tensor is needed.
    rank = math.ceil(0.99 * x.shape[0])
    parts = []
    for start in range(0, x.shape[1], 256):
        part = x[:, start:start + 256].abs().float()
        parts.append(torch.kthvalue(part, rank, dim=0).values)
    return torch.cat(parts).clamp_min(1e-5)


def main():
    a = parse_args()
    if a.calibration.resolve() == a.evaluation.resolve():
        raise ValueError('Calibration and evaluation must be separate calls')
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    c_cpu, c_call = load_input(a.calibration)
    e_cpu, e_call = load_input(a.evaluation)
    if c_call == e_call:
        raise ValueError('Calibration and evaluation cached calls overlap')
    if a.max_rows:
        c_cpu, e_cpu = c_cpu[:a.max_rows], e_cpu[:a.max_rows]
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(KEY)
    if w_cpu.shape != (5376, 14336) or w_cpu.dtype != torch.bfloat16:
        raise ValueError('Unexpected FC2 BF16 weight')
    if c_cpu.shape[1] != w_cpu.shape[1] or e_cpu.shape[1] != w_cpu.shape[1]:
        raise ValueError('K dimension mismatch')
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        c = c_cpu.to(device)
        e = e_cpu.to(device)
        w = w_cpu.to(device)
        del c_cpu, e_cpu, w_cpu
        levels = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=device)
        activation_channel_max = activation_channel_stat(c, a.activation_stat)
        weight_channel_max = w.abs().amax(dim=0).float().clamp_min(1e-5)
        if not bool(torch.isfinite(activation_channel_max).all()) or not bool(torch.isfinite(weight_channel_max).all()):
            raise ValueError('Nonfinite channel maxima')
        c_reference = c @ w.T
        e_reference = e @ w.T
        denominator = {
            'calibration': c_reference.float().square().sum(),
            'evaluation': e_reference.float().square().sum(),
        }
        results = []
        for alpha in (None, *ALPHAS):
            if alpha is None:
                scale = torch.ones_like(weight_channel_max)
                label = 'none'
            else:
                scale = (activation_channel_max.pow(alpha) /
                         weight_channel_max.pow(1.0 - alpha)).clamp_min(1e-5)
                label = str(alpha)
            if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
                raise ValueError(f'Invalid smoothing scale for {label}')
            # BF16 casts reflect the actual native NVFP4 quantizer input contract.
            cs = (c.float() / scale).to(torch.bfloat16)
            es = (e.float() / scale).to(torch.bfloat16)
            ws = (w.float() * scale).to(torch.bfloat16)
            c_bf16 = cs @ ws.T
            e_bf16 = es @ ws.T
            c_amax = float(cs.abs().amax().item())
            e_amax = float(es.abs().amax().item())
            w_amax = float(ws.abs().amax().item())
            if not all(math.isfinite(v) and v > 0 for v in (c_amax, e_amax, w_amax)):
                raise ValueError(f'Invalid smoothed amax for {label}')
            a_inverse = torch.tensor([2688.0 / c_amax], device=device, dtype=torch.float32)
            w_inverse = torch.tensor([2688.0 / w_amax], device=device, dtype=torch.float32)
            wq, wsf = nvfp4_quantize(ws, w_inverse, sfLayout=SfLayout.layout_128x4,
                                     backend='cuda')
            row = dict(smoothquant_alpha=alpha, label=label,
                       activation_global_amax_from_calibration=c_amax,
                       evaluation_activation_amax=e_amax,
                       evaluation_to_calibration_activation_amax_ratio=e_amax / c_amax,
                       weight_global_amax=w_amax,
                       scale_stats=dict(min=float(scale.min().item()),
                                        median=float(scale.median().item()),
                                        max=float(scale.max().item())),
                       weight_qdq_nmse=qdq_error(ws, w_inverse, levels,
                                                  nvfp4_quantize, SfLayout.layout_linear),
                       calls={})
            for name, x, xs, ref, bf16 in (
                ('calibration', c, cs, c_reference, c_bf16),
                ('evaluation', e, es, e_reference, e_bf16),
            ):
                aq, asf = nvfp4_quantize(xs, a_inverse, sfLayout=SfLayout.layout_128x4,
                                         backend='cuda')
                y = mm_fp4(aq, wq.T, asf, wsf.T,
                           alpha=(a_inverse * w_inverse).reciprocal(),
                           out_dtype=torch.bfloat16, backend='cutlass')
                output_nmse = float(((y.float() - ref.float()).square().sum() /
                                     denominator[name]).item())
                row['calls'][name] = dict(
                    output_nmse=output_nmse,
                    bf16_smoothing_roundoff_nmse=squared_error_ratio(bf16, ref),
                    activation_qdq_nmse=qdq_error(xs, a_inverse, levels,
                                                   nvfp4_quantize, SfLayout.layout_linear),
                )
                del aq, asf, y
            # Diagnostic: keep smoothing fixed, but derive only A's global
            # scale from the current evaluation input. Never select alpha on it.
            e_inverse = torch.tensor([2688.0 / e_amax], device=device, dtype=torch.float32)
            eq, esf = nvfp4_quantize(es, e_inverse, sfLayout=SfLayout.layout_128x4,
                                     backend='cuda')
            y_dynamic = mm_fp4(eq, wq.T, esf, wsf.T,
                               alpha=(e_inverse * w_inverse).reciprocal(),
                               out_dtype=torch.bfloat16, backend='cutlass')
            row['calls']['evaluation']['dynamic_global_output_nmse'] = float(
                ((y_dynamic.float() - e_reference.float()).square().sum() /
                 denominator['evaluation']).item())
            row['calls']['evaluation']['dynamic_global_activation_qdq_nmse'] = qdq_error(
                es, e_inverse, levels, nvfp4_quantize, SfLayout.layout_linear)
            del eq, esf, y_dynamic
            results.append(row)
            print(f'alpha={label:>4} calib={row["calls"]["calibration"]["output_nmse"]:.8f} '
                  f'eval={row["calls"]["evaluation"]["output_nmse"]:.8f}', flush=True)
            del cs, es, ws, c_bf16, e_bf16, wq, wsf

        expected = None if a.max_rows else expected_baselines()
        if expected is not None:
            observed = results[0]['calls']
            for name in ('calibration', 'evaluation'):
                delta = abs(observed[name]['output_nmse'] - expected[name])
                if delta > 2e-7:
                    raise RuntimeError(f'Unsmoothed {name} baseline failed E089 gate: {delta}')
            dynamic_delta = abs(observed['evaluation']['dynamic_global_output_nmse'] -
                                expected['evaluation_dynamic_global'])
            if dynamic_delta > 2e-7:
                raise RuntimeError(f'Unsmoothed dynamic-global baseline failed E089 gate: {dynamic_delta}')
        baseline = results[0]['calls']
        for row in results:
            for name in ('calibration', 'evaluation'):
                row['calls'][name]['relative_output_nmse_to_unsmoothed'] = (
                    row['calls'][name]['output_nmse'] / baseline[name]['output_nmse'])
        best_on_calibration = min(results[1:],
                                  key=lambda r: r['calls']['calibration']['output_nmse'])
        report = dict(format='h3-smoothquant-nvfp4-fc2-error-v1',
                      timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      scope='one H3 block-25 FC2, two real calls, error only',
                      calibration=str(a.calibration), evaluation=str(a.evaluation),
                      calibration_call=c_call, evaluation_call=e_call,
                      weight=str(WEIGHT), weight_key=KEY,
                      shape=dict(calibration_M=c.shape[0], evaluation_M=e.shape[0],
                                 K=w.shape[1], N=w.shape[0]),
                      smoke_subset=bool(a.max_rows), gpu=a.gpu,
                      activation_stat=a.activation_stat,
                      smoothing=f's_j = {a.activation_stat}_calibration|A_:,j|^alpha / max|W_:,j|^(1-alpha); A/s, W*s; clamp minima 1e-5',
                      quantization='FlashInfer native NVFP4 E2M1, E4M3 per K-16 block, per-tensor FP32 global scale from smoothed calibration A and smoothed W',
                      alpha_selection='minimum calibration-call output NMSE; no evaluation-call tuning',
                      dynamic_global_control='diagnostic only: same smoothing and W; A global scale from current evaluation call',
                      expected_unsmoothed_nmse=expected,
                      calibration_selected_alpha=best_on_calibration['smoothquant_alpha'],
                      rows=results)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'wrote {a.output}', flush=True)


if __name__ == '__main__':
    main()
