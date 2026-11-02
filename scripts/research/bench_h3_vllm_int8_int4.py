#!/usr/bin/env python3
"""H3 FC2 deployment-operator probe: vLLM Triton W8A8 and Marlin W4A16.

This benchmarks one real linear call, not a vLLM model engine or DiT block.
The vLLM CUDA kernels are used directly. Weight quantization/repacking is
offline; dynamic activation quantization is included in W8A8 online latency.
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
INPUT_KEY = 'mlp.fc2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=22400,
                        help='Prefix of real packed tokens; full call is 22400')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--batch-repeats', type=int, default=30)
    parser.add_argument('--batches', type=int, default=9)
    parser.add_argument('--arm', choices=['both', 'w8a8', 'w4a16'], default='both')
    return parser.parse_args()


def load_input(path: Path, tokens: int):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('format') != 'h3-native-bench-sample-v1' or payload.get('target_block') != 25:
        raise ValueError(f'{path}: not the registered H3 block-25 sample')
    x = payload['linear_inputs'][INPUT_KEY]
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (22400, 14336):
        raise ValueError(f'Unexpected FC2 input {x.dtype}, {tuple(x.shape)}')
    if not 1 <= tokens <= x.shape[0]:
        raise ValueError('tokens must be in [1, 22400]')
    return x[:tokens].contiguous(), str(payload['cache_call'])


def timed(fn, warmup: int, repeats: int, batches: int):
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


def error_metrics(actual: torch.Tensor, reference: torch.Tensor):
    if not bool(torch.isfinite(actual).all()):
        raise RuntimeError('Non-finite kernel output')
    numerator = (actual.float() - reference.float()).square().sum()
    denominator = reference.float().square().sum()
    nmse = float(numerator / denominator)
    return dict(nmse=nmse, relative_rms_error=math.sqrt(nmse),
                cosine=float(torch.nn.functional.cosine_similarity(
                    actual.float().reshape(1, -1),
                    reference.float().reshape(1, -1)).item()))


def w8a8_arm(x, w, reference, ops, a):
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm)

    # SM120 is rejected by vLLM's CUTLASS INT8 dispatch; this is the actual
    # vLLM Triton INT8 fallback. Weight is offline packed as [K,N], with
    # K-contiguous/column-major stride and a symmetric channel scale.
    scale_w = w.float().abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q_weight = torch.round(w.float() / scale_w[:, None]).clamp(-127, 127).to(torch.int8)
    q_weight = q_weight.T
    scale_w = scale_w.reshape(-1, 1).contiguous()

    def quantize():
        return ops.scaled_int8_quant(x.contiguous(), symmetric=True)

    def gemm(q_x, scale_x):
        return triton_scaled_mm(q_x, q_weight, scale_a=scale_x,
                                scale_b=scale_w, out_dtype=torch.bfloat16)

    def full():
        q_x, scale_x, azp = quantize()
        if azp is not None:
            raise RuntimeError('Symmetric INT8 quantization produced zero points')
        return gemm(q_x, scale_x)

    y = full()
    metrics = error_metrics(y, reference)
    q_x, scale_x, _ = quantize()
    if q_x.dtype != torch.int8 or scale_x.shape != (x.shape[0], 1):
        raise RuntimeError('Unexpected vLLM per-token INT8 contract')
    results = dict(quantization='symmetric per-token A8 / per-output-channel W8',
                   online_ops=['vllm.scaled_int8_quant', 'vllm.triton_scaled_mm'],
                   weight_scale_shape=list(scale_w.shape),
                   weight_stride=list(q_weight.stride()),
                   output=metrics,
                   timing={'activation_quant': timed(quantize, a.warmup,
                                                     a.batch_repeats, a.batches),
                           'gemm_prepacked': timed(lambda: gemm(q_x, scale_x),
                                                   a.warmup, a.batch_repeats, a.batches),
                           'full': timed(full, a.warmup, a.batch_repeats, a.batches)})
    return results


def w4a16_arm(x, w, reference, ops, scalar_types, a):
    from vllm.model_executor.layers.quantization.utils.quant_utils import pack_rows

    # Offline GPTQ-layout uint4b8 packing and the same group-scale
    # permutation as vLLM marlin_utils. No activation-order or Hessian-based
    # GPTQ weight correction is applied; the online operator is unchanged.
    k, n = w.shape[1], w.shape[0]
    group_size = 128
    groups = w.T.contiguous().reshape(k // group_size, group_size, n).float()
    scales = torch.maximum(groups.amax(dim=1) / 7.0,
                           -groups.amin(dim=1) / 8.0).clamp_min(1e-12)
    codes = (torch.round(groups / scales[:, None, :]).clamp(-8, 7) + 8).to(torch.int32)
    q_ref_weight = ((codes.float() - 8) * scales[:, None, :]).reshape(k, n).to(w.dtype)
    gptq_packed = pack_rows(codes.reshape(k, n).contiguous(), 4, k, n)
    packed = ops.gptq_marlin_repack(
        gptq_packed, torch.empty(0, dtype=torch.int32, device=x.device), k, n, 4)
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scales = scales.reshape(-1, 64)[:, scale_perm].reshape(k // group_size, n)
    scales = scales.to(torch.bfloat16).contiguous()
    workspace = torch.zeros((n // 64) * 16, dtype=torch.int32, device=x.device)
    del groups, codes, gptq_packed

    def full():
        return ops.marlin_gemm(
            x, None, packed, None, scales, None, None, None, None, None,
            workspace, scalar_types.uint4b8, size_m=x.shape[0], size_n=n,
            size_k=k, is_k_full=True, use_atomic_add=False,
            use_fp32_reduce=True, is_zp_float=False)

    y = full()
    metrics = error_metrics(y, reference)
    # Numeric layout gate: Marlin should reproduce its own dequantized weight
    # reference much more closely than the INT4-vs-BF16 quantization error.
    gate_tokens = min(16, x.shape[0])
    y_weight_ref = x[:gate_tokens] @ q_ref_weight
    layout_gate = error_metrics(y[:gate_tokens], y_weight_ref)
    if layout_gate['nmse'] > 1e-3:
        raise RuntimeError(f'Marlin weight layout gate failed: {layout_gate}')
    return dict(quantization='symmetric uint4b8, K-group 128, round-to-nearest; no GPTQ correction',
                online_ops=['vllm.marlin_gemm'],
                packed_shape=list(packed.shape), scale_shape=list(scales.shape),
                layout_gate=layout_gate, output=metrics,
                timing={'full': timed(full, a.warmup, a.batch_repeats, a.batches)})


def main():
    a = parse_args()
    if a.warmup < 1 or a.batch_repeats < 1 or a.batches < 3:
        raise ValueError('Insufficient timing repetitions')
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    import vllm

    x_cpu, call = load_input(a.sample, a.tokens)
    with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
        w_cpu = f.get_tensor(WEIGHT_KEY)
    if w_cpu.dtype != torch.bfloat16 or tuple(w_cpu.shape) != (5376, 14336):
        raise ValueError('Unexpected FC2 BF16 weight')
    device = torch.device(f'cuda:{a.gpu}')
    torch.cuda.set_device(device)
    with torch.inference_mode():
        x, w = x_cpu.to(device), w_cpu.to(device)
        reference = x @ w.T
        results = {}
        if a.arm in ('both', 'w8a8'):
            results['w8a8'] = w8a8_arm(x, w, reference, ops, a)
        if a.arm in ('both', 'w4a16'):
            results['w4a16'] = w4a16_arm(x, w, reference, ops, scalar_types, a)
        bf16_timing = timed(lambda: x @ w.T, a.warmup, a.batch_repeats, a.batches)

    report = dict(format='h3-vllm-int8-int4-linear-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  scope='one H3 FC2 linear, not a vLLM engine or full DiT block',
                  vllm_version=vllm.__version__, torch_version=torch.__version__,
                  gpu=a.gpu, gpu_name=torch.cuda.get_device_name(device),
                  sample=str(a.sample), cached_call=call,
                  weight=str(WEIGHT), weight_key=WEIGHT_KEY,
                  shape=dict(M=a.tokens, K=14336, N=5376),
                  offline_weight_pack_excluded=True,
                  bf16_timing=bf16_timing, arms=results,
                  stable_cv_threshold_percent=3.0,
                  timing_stable=all(t['cv_percent'] <= 3.0 for r in results.values()
                                    for t in r['timing'].values()) and
                                bf16_timing['cv_percent'] <= 3.0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
