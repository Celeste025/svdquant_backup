#!/usr/bin/env python3
"""E092: H3 FC2 error-only signed INT4 W4A4 smoothing/granularity probe.

Quantized operands are decoded before BF16 GEMM; this is a numeric error
experiment, not a native INT4 kernel or speed measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors import safe_open

from eval_h3_nvfp4_row_scales import INPUT_KEY, KEY, WEIGHT, load_input
from eval_h3_smoothquant_nvfp4_fc2 import activation_channel_stat


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
GROUPS = (16, 128)


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--max-rows', type=int, default=0,
                   help='smoke subset, not reportable full result')
    return p.parse_args()


def int4_qdq(x: torch.Tensor, group: int):
    """Per-row contiguous K-group signed [-8,7] INT4 with FP32 scales."""
    if x.ndim != 2 or x.shape[1] % group:
        raise ValueError(f'Expected [rows,K] with K divisible by {group}')
    rows, k = x.shape
    xf = x.float().reshape(rows, k // group, group)
    scale = torch.maximum(xf.amax(dim=-1) / 7.0,
                          -xf.amin(dim=-1) / 8.0).clamp_min(1e-12)
    codes = torch.round(xf / scale[..., None]).clamp(-8, 7).to(torch.int8)
    decoded = (codes.float() * scale[..., None]).reshape(rows, k).to(torch.bfloat16)
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError('Nonfinite INT4 decoded operand')
    return decoded, codes, scale


def nmse(actual, ref, denominator):
    if not bool(torch.isfinite(actual).all()):
        raise RuntimeError('Nonfinite INT4 output')
    return float(((actual.float() - ref.float()).square().sum() / denominator).item())


def integer_code_gate(qx, sx, qw, sw, decoded_x, decoded_w):
    """Check QDQ BF16 GEMM against integer-code products on eight real rows."""
    take, groups = min(8, qx.shape[0]), qx.shape[1]
    exact = torch.zeros((take, qw.shape[0]), device=qx.device, dtype=torch.float32)
    for g in range(groups):
        products = qx[:take, g].float() @ qw[:, g].float().T
        exact += products * (sx[:take, g, None] * sw[None, :, g])
    proxy = decoded_x[:take] @ decoded_w.T
    return nmse(proxy, exact, exact.square().sum().clamp_min(1e-30))


def main():
    a = args()
    if a.calibration.resolve() == a.evaluation.resolve():
        raise ValueError('Calibration and evaluation must be separate calls')
    c_cpu, c_call = load_input(a.calibration)
    e_cpu, e_call = load_input(a.evaluation)
    if c_call == e_call:
        raise ValueError('Calibration/evaluation cached calls overlap')
    if a.max_rows:
        c_cpu, e_cpu = c_cpu[:a.max_rows], e_cpu[:a.max_rows]
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(KEY)
    if tuple(w_cpu.shape) != (5376, 14336) or w_cpu.dtype != torch.bfloat16:
        raise ValueError('Unexpected H3 FC2 BF16 weight')
    torch.cuda.set_device(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{a.gpu}')
    with torch.inference_mode():
        c, e, w = c_cpu.to(device), e_cpu.to(device), w_cpu.to(device)
        del c_cpu, e_cpu, w_cpu
        c_ref, e_ref = c @ w.T, e @ w.T
        den = (c_ref.float().square().sum(), e_ref.float().square().sum())
        w_stat = w.abs().amax(dim=0).float().clamp_min(1e-5)
        x_stats = {kind: activation_channel_stat(c, kind)
                   for kind in ('absmax', 'abs-p99')}
        if not all(bool(torch.isfinite(v).all()) for v in (*x_stats.values(), w_stat)):
            raise ValueError('Nonfinite smoothing statistics')
        rows = []
        for group in GROUPS:
            for kind in ('none', 'absmax', 'abs-p99'):
                for alpha in ((None,) if kind == 'none' else ALPHAS):
                    scale = (torch.ones_like(w_stat) if alpha is None else
                             (x_stats[kind].pow(alpha) / w_stat.pow(1.0 - alpha))
                             .clamp_min(1e-5))
                    if not bool(torch.isfinite(scale).all()):
                        raise ValueError(f'Nonfinite smoothing scale {group} {kind} {alpha}')
                    cs = (c.float() / scale).to(torch.bfloat16)
                    es = (e.float() / scale).to(torch.bfloat16)
                    ws = (w.float() * scale).to(torch.bfloat16)
                    w_dec, w_codes, w_scales = int4_qdq(ws, group)
                    results = {}
                    gate = None
                    for label, x, ref, d in (('calibration', cs, c_ref, den[0]),
                                             ('evaluation', es, e_ref, den[1])):
                        x_dec, x_codes, x_scales = int4_qdq(x, group)
                        y = x_dec @ w_dec.T
                        results[label] = dict(output_nmse=nmse(y, ref, d))
                        if label == 'calibration' and kind == 'none':
                            gate = integer_code_gate(x_codes, x_scales, w_codes,
                                                     w_scales, x_dec, w_dec)
                            if gate >= 2e-4:
                                raise RuntimeError(f'INT4 code versus BF16 QDQ gate failed: {group} {gate}')
                        del x_dec, x_codes, x_scales, y
                    row = dict(group_size=group, activation_stat=kind, alpha=alpha,
                               scale_stats=dict(min=float(scale.min()),
                                                median=float(scale.median()),
                                                max=float(scale.max())),
                               calls=results)
                    if gate is not None:
                        row['int4_code_vs_qdq_bf16_gate_nmse'] = gate
                    rows.append(row)
                    print(f'g={group:3} {kind:8} alpha={str(alpha):>4} '
                          f'p11={results["calibration"]["output_nmse"]:.8f} '
                          f'p1={results["evaluation"]["output_nmse"]:.8f}', flush=True)
                    del cs, es, ws, w_dec, w_codes, w_scales, scale
        selected = {}
        for group in GROUPS:
            selected[str(group)] = {}
            for kind in ('absmax', 'abs-p99'):
                best = min((r for r in rows if r['group_size'] == group and
                            r['activation_stat'] == kind),
                           key=lambda r: r['calls']['calibration']['output_nmse'])
                selected[str(group)][kind] = best['alpha']
        report = dict(format='h3-fc2-int4-group-smoothing-v1',
                      timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      scope='one H3 FC2, two real calls, error only; not native INT4 timing',
                      calibration=str(a.calibration), evaluation=str(a.evaluation),
                      calibration_call=c_call, evaluation_call=e_call,
                      weight=str(WEIGHT), weight_key=KEY, gpu=a.gpu,
                      shape=dict(calibration_M=c.shape[0], evaluation_M=e.shape[0],
                                 K=w.shape[1], N=w.shape[0]),
                      smoke_subset=bool(a.max_rows),
                      quantization='signed [-8,7] INT4, round-nearest, per-row contiguous K group; FP32 scales; decoded BF16 GEMM, BF16 output',
                      smoothing='s_j = (calibration per-channel absmax or nearest-rank abs-p99)^alpha / weight per-input-channel absmax^(1-alpha)',
                      alpha_selection='minimum p11 calibration output NMSE, independent for statistic/group; never p1',
                      selected_alpha=selected, rows=rows)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'wrote {a.output}', flush=True)


if __name__ == '__main__':
    main()
