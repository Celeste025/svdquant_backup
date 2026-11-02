#!/usr/bin/env python3
"""H3 FC2 W4A4: vLLM-style calibrated global scale + native NVFP4 kernels.

This is a *linear* benchmark, not a vLLM engine or DiT latency benchmark.
It uses FlashInfer's production quantizer and CUTLASS GEMM with the same
precomputed global-scale / online block-16 activation contract as vLLM's
FlashInferCutlassNvFp4LinearKernel. Weight conversion is offline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import torch
from safetensors import safe_open

WEIGHT = Path('/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/'
              'diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors')
KEY = 'blocks.25.mlp.fc2.weight'
INPUT_KEY = 'mlp.fc2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=Path, required=True, help='held-out real block replay')
    p.add_argument('--calibration', type=Path, required=True, help='different cached call')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=4)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--batch-repeats', type=int, default=40)
    p.add_argument('--batches', type=int, default=11)
    p.add_argument('--backend', choices=['cutlass', 'auto'], default='cutlass')
    return p.parse_args()


def sample_input(path):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != 'h3-native-bench-sample-v1' or payload.get('target_block') != 25:
        raise ValueError(f'{path}: unexpected H3 sample contract')
    x = payload['linear_inputs'][INPUT_KEY]
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] != 14336:
        raise ValueError(f'{path}: unexpected FC2 input {x.dtype} {tuple(x.shape)}')
    return x, str(payload['cache_call'])


def timing(fn, warmup, batch_repeats, batches):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(batches):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch_repeats):
            fn()
        stop.record()
        stop.synchronize()
        measurements.append(start.elapsed_time(stop) / batch_repeats)
    mean = statistics.mean(measurements)
    return dict(median_ms=statistics.median(measurements), mean_ms=mean,
                cv_percent=100 * statistics.stdev(measurements) / mean if batches > 1 else 0,
                min_ms=min(measurements), max_ms=max(measurements), batch_ms=measurements)


def main():
    a = parse_args()
    if a.sample.resolve() == a.calibration.resolve():
        raise ValueError('Calibration and evaluation samples must be distinct files')
    if a.warmup < 1 or a.batch_repeats < 1 or a.batches < 3:
        raise ValueError('Insufficient timing repetitions')
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    import flashinfer
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    x_cpu, eval_call = sample_input(a.sample)
    c_cpu, calib_call = sample_input(a.calibration)
    if eval_call == calib_call:
        raise ValueError('Calibration and evaluation cached calls must be distinct')
    if x_cpu.shape != (22400, 14336):
        raise ValueError(f'Expected original full-shape FC2 input, got {tuple(x_cpu.shape)}')
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(KEY)
    if w_cpu.shape != (5376, 14336) or w_cpu.dtype != torch.bfloat16:
        raise ValueError('Unexpected BF16 FC2 checkpoint weight')

    # CPU-side offline scales; no evaluation activation contributes to a_global.
    calibration_amax = float(c_cpu.float().abs().amax())
    evaluation_amax = float(x_cpu.float().abs().amax())
    weight_amax = float(w_cpu.float().abs().amax())
    if not all(math.isfinite(z) and z > 0 for z in
               (calibration_amax, evaluation_amax, weight_amax)):
        raise ValueError('Invalid calibration or weight amax')
    global_a = 448.0 * 6.0 / calibration_amax
    global_w = 448.0 * 6.0 / weight_amax
    device = torch.device(f'cuda:{a.gpu}')
    torch.cuda.set_device(device)
    with torch.inference_mode():
        x = x_cpu.to(device, non_blocking=False)
        w = w_cpu.to(device, non_blocking=False)
        a_scale = torch.tensor([global_a], dtype=torch.float32, device=device)
        w_scale = torch.tensor([global_w], dtype=torch.float32, device=device)
        alpha = (a_scale * w_scale).reciprocal()
        # Offline weight pack is deliberately outside steady-state timing.
        wq, wsf = nvfp4_quantize(w, w_scale, sfLayout=SfLayout.layout_128x4,
                                 do_shuffle=False, backend='cuda')
        wq_t, wsf_t = wq.T, wsf.T
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=device)
        def quantize():
            return nvfp4_quantize(x, a_scale, sfLayout=SfLayout.layout_128x4,
                                   do_shuffle=False, backend='cuda')
        def gemm(q, sf):
            return mm_fp4(q, wq_t, sf, wsf_t, alpha=alpha, out_dtype=torch.bfloat16,
                          out=out, backend=a.backend)
        def full():
            q, sf = quantize()
            return gemm(q, sf)
        def bf16():
            return torch.mm(x, w.T, out=out)

        # Correctness is checked before timing; BF16 reference is not reused as
        # an output buffer because the quantized GEMM overwrites that buffer.
        reference = torch.mm(x, w.T)
        y = full().clone()
        if not bool(torch.isfinite(y).all()):
            raise RuntimeError('NVFP4 output contains non-finite values')
        diff = y.float() - reference.float()
        nmse = float(diff.square().sum() / reference.float().square().sum())
        cos = float(torch.nn.functional.cosine_similarity(
            y.float().flatten().unsqueeze(0),
            reference.float().flatten().unsqueeze(0)).item())
        # Accuracy-only control: same kernels and packed weight, but a global
        # scale derived from this evaluation input. It is never timed or used
        # as the calibrated-scale deployment result.
        eval_scale = torch.tensor([448.0 * 6.0 / evaluation_amax],
                                  dtype=torch.float32, device=device)
        eq, esf = nvfp4_quantize(x, eval_scale, sfLayout=SfLayout.layout_128x4,
                                  do_shuffle=False, backend='cuda')
        dynamic_alpha = (eval_scale * w_scale).reciprocal()
        y_dynamic = mm_fp4(eq, wq_t, esf, wsf_t, alpha=dynamic_alpha,
                           out_dtype=torch.bfloat16, out=out, backend=a.backend)
        dynamic_nmse = float((y_dynamic.float() - reference.float()).square().sum()
                             / reference.float().square().sum())
        del eq, esf, y_dynamic
        del y, diff, reference
        q, sf = quantize()
        actual_kernel = dict(codes_shape=list(q.shape), codes_dtype=str(q.dtype),
                             scales_shape=list(sf.shape), scales_dtype=str(sf.dtype))
        torch.cuda.synchronize()
        measurements = {
            'bf16': timing(bf16, a.warmup, a.batch_repeats, a.batches),
            'nvfp4_quantize': timing(quantize, a.warmup, a.batch_repeats, a.batches),
            'nvfp4_gemm_prepacked': timing(lambda: gemm(q, sf), a.warmup, a.batch_repeats, a.batches),
            'nvfp4_full': timing(full, a.warmup, a.batch_repeats, a.batches),
        }

    report = dict(format='h3-vllm-style-nvfp4-linear-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='one H3 blocks.25.mlp.fc2, not a model engine',
                  device=torch.cuda.get_device_name(device), gpu=a.gpu,
                  torch_version=torch.__version__, flashinfer_version=flashinfer.__version__,
                  backend=a.backend, sample=str(a.sample), calibration=str(a.calibration),
                  evaluation_call=eval_call, calibration_call=calib_call,
                  weight=str(WEIGHT), weight_key=KEY,
                  shape=dict(M=x.shape[0], K=x.shape[1], N=w.shape[0]),
                  calibration_amax=calibration_amax, evaluation_amax=evaluation_amax,
                  weight_amax=weight_amax,
                  input_global_scale=global_a, weight_global_scale=global_w,
                  quantized_contract=actual_kernel, output_nmse=nmse,
                  output_relative_rms_error=math.sqrt(nmse), output_cosine=cos,
                  eval_derived_scale_accuracy_control_nmse=dynamic_nmse,
                  measurements=measurements,
                  validity=dict(calibration_disjoint=True,
                                weight_pack_excluded=True,
                                online_block16_quant_included=True,
                                external_global_amax_excluded=True,
                                bias='none in original FC2',
                                stable_cv_threshold_percent=3.0,
                                stable=all(row['cv_percent'] <= 3.0
                                           for row in measurements.values())))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
