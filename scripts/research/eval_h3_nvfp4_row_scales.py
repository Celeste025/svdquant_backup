#!/usr/bin/env python3
"""Error-only H3 W4A4 NVFP4 comparison of FP32 global-scale row groups.

Native FlashInfer quantizes each group. The actual FP4 codes and E4M3 block
scales are dequantized before a FP32 reference GEMM because the installed
dense NVFP4 GEMM accepts only a scalar global alpha. A native per-tensor
NVFP4 GEMM gates the dequantized comparison before results are accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


WEIGHT = Path('/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/'
              'diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors')
KEY = 'blocks.25.mlp.fc2.weight'
INPUT_KEY = 'mlp.fc2'
ROW_GROUPS = (512, 128, 32, 1)
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=7)
    p.add_argument('--max-rows', type=int, default=0,
                   help='positive value runs a smoke subset, not a reportable full call')
    return p.parse_args()


def load_input(path):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != 'h3-native-bench-sample-v1' or payload.get('target_block') != 25:
        raise ValueError(f'Unexpected sample contract: {path}')
    x = payload['linear_inputs'][INPUT_KEY]
    if x.ndim != 2 or x.shape[1] != 14336 or x.dtype != torch.bfloat16:
        raise ValueError(f'Unexpected FC2 input: {x.shape}, {x.dtype}')
    return x, str(payload['cache_call'])


def dequantize(q, sf, global_descale, levels):
    """Decode row-major K-contiguous NVFP4 bytes and linear E4M3 scales."""
    packed = q.view(torch.uint8)
    if packed.ndim != 2 or sf.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ValueError(f'Unexpected linear NVFP4 layout: {packed.shape}, {sf.shape}')
    scale = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    scale = scale * global_descale
    out = torch.empty((packed.shape[0], packed.shape[1] * 2),
                      dtype=torch.float32, device=packed.device)
    out[:, 0::2] = levels[(packed & 15).long()] * scale[:, 0::2]
    out[:, 1::2] = levels[(packed >> 4).long()] * scale[:, 1::2]
    if not bool(torch.isfinite(out).all()):
        raise ValueError('Decoded NVFP4 contains non-finite values')
    return out


def group_max(x, rows):
    full = x.shape[0] // rows
    result = x[:full * rows].abs().reshape(full, rows, x.shape[1]).amax(dim=(1, 2)).float()
    if full * rows < x.shape[0]:
        result = torch.cat((result, x[full * rows:].abs().amax().float().reshape(1)))
    if not bool((result > 0).all()) or not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid row-group amax')
    return result


def quant_dequant(x, rows, levels, nvfp4_quantize, sf_layout, fixed_inverse=None):
    """Pack in K-blocks of 16, then decode. `rows=1` uses native per-token mode."""
    m, k = x.shape
    out = torch.empty((m, k), dtype=torch.float32, device=x.device)
    if fixed_inverse is not None:
        q, sf = nvfp4_quantize(x, fixed_inverse, sfLayout=sf_layout, backend='cuda')
        for start in range(0, m, 128):
            stop = min(start + 128, m)
            out[start:stop] = dequantize(q[start:stop], sf[start:stop],
                                         fixed_inverse.reciprocal(), levels)
        del q, sf
        return out
    if rows == 1:
        q, sf, row_descale = nvfp4_quantize(
            x, 1.0 / 2688.0, sfLayout=sf_layout, backend='cuda',
            per_token_activation=True)
        for start in range(0, m, 128):
            stop = min(start + 128, m)
            out[start:stop] = dequantize(q[start:stop], sf[start:stop],
                                         row_descale[start:stop, None], levels)
        del q, sf, row_descale
        return out
    inverse = 2688.0 / group_max(x, rows)
    for index, start in enumerate(range(0, m, rows)):
        stop = min(start + rows, m)
        q, sf = nvfp4_quantize(x[start:stop], inverse[index:index + 1],
                               sfLayout=sf_layout, backend='cuda')
        out[start:stop] = dequantize(q, sf, inverse[index].reciprocal(), levels)
    return out


def nmse(y, reference, denominator):
    difference = y.float() - reference.float()
    return float((difference.square().sum() / denominator).item())


def main():
    a = args()
    if a.sample.resolve() == a.calibration.resolve():
        raise ValueError('Calibration and evaluation files must differ')
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    x_cpu, sample_call = load_input(a.sample)
    c_cpu, calib_call = load_input(a.calibration)
    if sample_call == calib_call:
        raise ValueError('Calibration and evaluation cached calls must differ')
    if a.max_rows:
        x_cpu = x_cpu[:a.max_rows]
    if x_cpu.shape[1] % 16:
        raise ValueError('K is not a multiple of 16')
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(KEY)
    if w_cpu.shape != (5376, 14336) or w_cpu.dtype != torch.bfloat16:
        raise ValueError('Unexpected FC2 weight')
    calib_amax = float(c_cpu.abs().amax())
    if not math.isfinite(calib_amax) or calib_amax <= 0:
        raise ValueError('Invalid calibration amax')
    del c_cpu

    torch.cuda.set_device(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{a.gpu}')
    with torch.inference_mode():
        x = x_cpu.to(device)
        w = w_cpu.to(device)
        del x_cpu, w_cpu
        levels = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=device)
        a_inverse = torch.tensor([2688.0 / calib_amax], dtype=torch.float32, device=device)
        w_amax = float(w.abs().amax().item())
        w_inverse = torch.tensor([2688.0 / w_amax], dtype=torch.float32, device=device)
        # Native W4A4 baseline. Its block-scale layout is swizzled for Tensor Cores.
        aq, asf = nvfp4_quantize(x, a_inverse, sfLayout=SfLayout.layout_128x4,
                                 backend='cuda')
        wq, wsf = nvfp4_quantize(w, w_inverse, sfLayout=SfLayout.layout_128x4,
                                 backend='cuda')
        native = mm_fp4(aq, wq.T, asf, wsf.T,
                        alpha=(a_inverse * w_inverse).reciprocal(),
                        out_dtype=torch.bfloat16, backend='cutlass')
        del aq, asf, wq, wsf
        reference = x @ w.T
        denominator = reference.float().square().sum()
        baseline_native_nmse = nmse(native, reference, denominator)
        del reference
        # The same BF16 reference is re-created for results after the gate.
        w_deq = {}
        w_deq['tensor'] = quant_dequant(w, 0, levels, nvfp4_quantize,
                                        SfLayout.layout_linear, w_inverse)
        x_deq = {}
        x_deq['calibrated_tensor'] = quant_dequant(x, 0, levels, nvfp4_quantize,
                                                  SfLayout.layout_linear, a_inverse)
        gate_out = (x_deq['calibrated_tensor'] @ w_deq['tensor'].T).to(torch.bfloat16)
        gate_nmse = float(((gate_out.float() - native.float()).square().sum() /
                           native.float().square().sum()).item())
        del gate_out, native
        print(f'native gate: baseline NMSE={baseline_native_nmse:.8f}, '
              f'FP32 dequant-GEMM vs native NMSE={gate_nmse:.8g}', flush=True)
        if gate_nmse > 2e-5:
            raise RuntimeError('Native versus dequantized-GEMM gate failed')

        x_deq['eval_tensor'] = quant_dequant(x, 0, levels, nvfp4_quantize,
                                             SfLayout.layout_linear,
                                             2688.0 / x.abs().amax().float())
        for group in ROW_GROUPS:
            x_deq[str(group)] = quant_dequant(x, group, levels, nvfp4_quantize,
                                               SfLayout.layout_linear)
            w_deq[str(group)] = quant_dequant(w, group, levels, nvfp4_quantize,
                                               SfLayout.layout_linear)
            print(f'packed/dequantized row group {group}', flush=True)
        reference = x @ w.T
        rows = []

        def evaluate(a_key, b_key):
            output = (x_deq[a_key] @ w_deq[b_key].T).to(torch.bfloat16)
            value = nmse(output, reference, denominator)
            rows.append(dict(activation_scale=a_key, weight_scale=b_key,
                             output_nmse=value,
                             relative_to_calibrated_baseline=None))
            print(f'A={a_key:>17} W={b_key:>6} NMSE={value:.8f}', flush=True)
            del output

        evaluate('calibrated_tensor', 'tensor')
        evaluate('eval_tensor', 'tensor')
        for group in ROW_GROUPS:
            key = str(group)
            evaluate(key, 'tensor')
            evaluate('calibrated_tensor', key)
            evaluate(key, key)
        baseline = rows[0]['output_nmse']
        for row in rows:
            row['relative_to_calibrated_baseline'] = row['output_nmse'] / baseline
        result = dict(format='h3-nvfp4-row-scales-v1',
                      timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      scope='one real H3 block-25 FC2 linear; error only',
                      sample=str(a.sample), calibration=str(a.calibration),
                      sample_call=sample_call, calibration_call=calib_call,
                      weight=str(WEIGHT), weight_key=KEY,
                      shape=dict(M=x.shape[0], K=x.shape[1], N=w.shape[0]),
                      smoke_subset=bool(a.max_rows), gpu=a.gpu,
                      quantization='native FlashInfer NVFP4 E2M1 and E4M3 per K-16 block; grouped FP32 global scales',
                      comparison_gemm='FP32 GEMM on dequantized native codes and scales, output rounded BF16',
                      calibrated_activation_amax=calib_amax,
                      weight_amax=w_amax,
                      native_baseline_nmse=baseline_native_nmse,
                      proxy_vs_native_nmse=gate_nmse,
                      proxy_gate_threshold=2e-5,
                      rows=rows)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'wrote {a.output}', flush=True)


if __name__ == '__main__':
    main()
