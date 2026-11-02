#!/usr/bin/env python3
"""Native NVFP4-weight/BF16-activation H3 FC2 benchmark via FlashInfer.

Only the W4A16 BF16 x FP4 GEMM is online. Weight NVFP4 quantization and
backend-specific repacking are offline, as in a weight-only deployment.
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
WEIGHT_KEY = 'blocks.25.mlp.fc2.weight'


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--tokens', type=int, default=22400)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--batch-repeats', type=int, default=10)
    parser.add_argument('--batches', type=int, default=7)
    parser.add_argument('--backend', choices=['cute-dsl', 'cudnn'], default='cute-dsl')
    parser.add_argument('--cutlass-site', type=Path,
                        help='isolated CuTe DSL pip --target directory; processed via site.addsitedir')
    return parser.parse_args()


def load_input(path: Path, tokens: int):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != 'h3-native-bench-sample-v1' or payload.get('target_block') != 25:
        raise ValueError(f'Unexpected H3 sample contract: {path}')
    x = payload['linear_inputs']['mlp.fc2']
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (22400, 14336):
        raise ValueError(f'Unexpected FC2 input: {x.dtype}, {tuple(x.shape)}')
    if not 1 <= tokens <= 22400:
        raise ValueError('tokens must be in [1, 22400]')
    return x[:tokens].contiguous(), str(payload['cache_call'])


def timed(fn, warmup, repeats, batches):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(batches):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / repeats)
    mean = statistics.mean(samples)
    return dict(median_ms=statistics.median(samples), min_ms=min(samples),
                mean_ms=mean, cv_percent=100 * statistics.stdev(samples) / mean,
                batch_ms=samples)


def main():
    a = args()
    if a.warmup < 1 or a.batch_repeats < 1 or a.batches < 3:
        raise ValueError('Insufficient timing repetitions')
    if a.cutlass_site is not None:
        import site
        if not a.cutlass_site.is_dir():
            raise FileNotFoundError(a.cutlass_site)
        site.addsitedir(str(a.cutlass_site.resolve()))
    import flashinfer
    from flashinfer import SfLayout, mm_bf16_fp4, nvfp4_quantize

    x_cpu, call = load_input(a.sample, a.tokens)
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(WEIGHT_KEY)
    if w_cpu.dtype != torch.bfloat16 or tuple(w_cpu.shape) != (5376, 14336):
        raise ValueError('Unexpected BF16 FC2 weight')
    weight_amax = float(w_cpu.float().abs().amax())
    if not math.isfinite(weight_amax) or weight_amax <= 0:
        raise ValueError('Invalid weight max')
    # Match the weight-global-scale recipe of the existing NVFP4 W4A4 arm.
    global_w = 448.0 * 6.0 / weight_amax
    device = torch.device(f'cuda:{a.gpu}')
    torch.cuda.set_device(device)
    with torch.inference_mode():
        x, w = x_cpu.to(device), w_cpu.to(device)
        w_scale = torch.tensor([global_w], dtype=torch.float32, device=device)
        alpha = w_scale.reciprocal()
        # Everything above the `full` function is offline model-load work.
        wq, wsf = nvfp4_quantize(w, w_scale, sfLayout=SfLayout.layout_128x4,
                                 do_shuffle=False, backend='cuda')
        wp, sfp, alphap = flashinfer.prepare_bf16_fp4_weights(
            wq, wsf, alpha, backend=a.backend)
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=device)

        def full():
            return mm_bf16_fp4(x, wp, sfp, alphap, backend=a.backend,
                               out_dtype=torch.bfloat16, out=out)

        reference = x @ w.T
        y = full().clone()
        if not bool(torch.isfinite(y).all()):
            raise RuntimeError('Non-finite W4A16 output')
        numerator = (y.float() - reference.float()).square().sum()
        denominator = reference.float().square().sum()
        nmse = float(numerator / denominator)
        cosine = float(torch.nn.functional.cosine_similarity(
            y.float().reshape(1, -1), reference.float().reshape(1, -1)).item())
        del y, reference
        torch.cuda.synchronize()
        measures = {
            'bf16': timed(lambda: torch.mm(x, w.T, out=out), a.warmup,
                          a.batch_repeats, a.batches),
            'nvfp4_w4a16_gemm': timed(full, a.warmup, a.batch_repeats, a.batches),
        }

    report = dict(format='h3-flashinfer-nvfp4-w4a16-linear-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='one H3 FC2 linear, not complete DiT or model engine',
                  flashinfer_version=flashinfer.__version__,
                  torch_version=torch.__version__, gpu=a.gpu,
                  gpu_name=torch.cuda.get_device_name(device),
                  sample=str(a.sample), cached_call=call,
                  weight=str(WEIGHT), weight_key=WEIGHT_KEY,
                  shape=dict(M=a.tokens, K=14336, N=5376),
                  quantization='NVFP4 E2M1 weight; block-16 E4M3 scales and offline global scale; BF16 activation',
                  backend=a.backend, input_activation_quantization=False,
                  cutlass_site=str(a.cutlass_site) if a.cutlass_site else None,
                  weight_global_scale=global_w,
                  offline_weight_quant_and_prepare_excluded=True,
                  prepared_weight=dict(shape=list(wp.shape), dtype=str(wp.dtype),
                                       scale_shape=list(sfp.shape), scale_dtype=str(sfp.dtype)),
                  output_nmse=nmse, output_relative_rms_error=math.sqrt(nmse),
                  output_cosine=cosine, measurements=measures,
                  stable_cv_threshold_percent=3.0,
                  timing_stable=all(m['cv_percent'] <= 3.0 for m in measures.values()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
