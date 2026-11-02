#!/usr/bin/env python3
"""Benchmark native Sage attention APIs on a reconstructed real H3 block call."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

SAMPLE = Path('/data1/models/svdquant-wjq/results/E088_native_bench/h3_block25_p1_s00.pt')
WEIGHT = Path('/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/'
              'diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors')
# RTX PRO 5000 Sage3 benchmark baseline; historical original can be selected
# explicitly via --sage3-target.
SAGE3 = Path('/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path,
                   help='Real post-norm/RoPE QKV capture; otherwise use the historical long-sequence FC2 sample')
    p.add_argument('--sage2-target', type=Path)
    p.add_argument('--sage3-target', type=Path, default=SAGE3)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--batches', type=int, default=11)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--full-helper', action='store_true',
                   help='Also time the main segment plus the 16-row padding segment and output assembly')
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def timed(fn, warmup, repeats, batches):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(batches):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return dict(median_ms=statistics.median(samples),
                cv_percent=100 * statistics.stdev(samples) / statistics.mean(samples),
                batch_ms=samples)


def metrics(y, ref):
    if y.shape != ref.shape or y.dtype != torch.bfloat16 or not bool(torch.isfinite(y).all()):
        raise RuntimeError(f'Invalid attention output: {y.shape} {y.dtype}')
    # Chunk the reduction to avoid materializing multiple full-size FP32 copies.
    err, energy, dot, y_energy = 0., 0., 0., 0.
    for begin in range(0, y.shape[2], 1024):
        a = y[:, :, begin:begin + 1024].float()
        b = ref[:, :, begin:begin + 1024].float()
        err += float((a - b).square().sum())
        energy += float(b.square().sum())
        dot += float((a * b).sum())
        y_energy += float(a.square().sum())
    return dict(nmse=err / energy, relative_rms_error=math.sqrt(err / energy),
                cosine=dot / math.sqrt(y_energy * energy))


def profile_kernels(fn, calls):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        if event.self_device_time_total > 0:
            rows.append(dict(kernel=event.key, calls=event.count,
                             total_gpu_ms=event.self_device_time_total / 1000,
                             mean_gpu_ms=event.self_device_time_total / (1000 * event.count)))
    return sorted(rows, key=lambda row: row['total_gpu_ms'], reverse=True)


def main():
    a = parse_args()
    if a.batches < 3 or a.repeats < 1 or a.warmup < 1:
        raise ValueError('Insufficient timing repetitions')
    if a.sage2_target is not None:
        sys.path.insert(0, str(a.sage2_target))
    sys.path.insert(0, str(Path('/home/wjq/workspace/DiffSynth-Studio')))
    if not (a.sage3_target / 'fp4attn_cuda.cpython-312-x86_64-linux-gnu.so').is_file():
        raise FileNotFoundError(f'Sage3 extension missing: {a.sage3_target}')
    sys.path.insert(0, str(a.sage3_target))
    from diffsynth.models.minimax_h3_dit import _apply_rope
    from sageattn3 import sageattn3_blackwell
    import sageattn3.api as sage3_api
    if not Path(sage3_api.__file__).resolve().is_relative_to(a.sage3_target.resolve()):
        raise RuntimeError(f'Sage3 imported from unexpected source: {sage3_api.__file__}')

    sample_path = a.capture or SAMPLE
    payload = torch.load(sample_path, map_location='cpu', weights_only=True, mmap=True)
    if a.capture:
        if payload['format'] != 'h3-480p3s-real-attention-v1' or payload['block'] != 25:
            raise ValueError('Unexpected real H3 attention capture')
        bounds = payload['cu_seqlens'].tolist()
        if bounds[0] != 0 or len(bounds) != 3 or bounds[-1] != payload['q'].shape[0]:
            raise ValueError(f'Unexpected captured H3 attention segments: {bounds}')
        if any(tuple(payload[name].shape) != (bounds[-1], 56, 128)
               for name in ('q', 'k', 'v')):
            raise ValueError('Unexpected captured QKV shape')
    else:
        if payload['format'] != 'h3-native-bench-sample-v1' or payload['target_block'] != 25:
            raise ValueError('Unexpected cached H3 sample')
        bounds = payload['block_kwargs']['cu_seqlens'].tolist()
        if bounds != [0, 22384, 22400]:
            raise ValueError(f'Unexpected H3 attention segments: {bounds}')
        with safe_open(str(WEIGHT), framework='pt', device='cpu') as f:
            w = f.get_tensor('blocks.25.attn.qkv_proj.weight')
            q_weight = f.get_tensor('blocks.25.attn.q_norm.weight')
            k_weight = f.get_tensor('blocks.25.attn.k_norm.weight')
    torch.cuda.set_device(a.gpu)
    device = torch.device(f'cuda:{a.gpu}')
    with torch.inference_mode():
        if a.capture:
            q, k, v = (payload[name].to(device) for name in ('q', 'k', 'v'))
            expected = None
        else:
            x = payload['linear_inputs']['attn.qkv_proj'].to(device)
            rope = payload['block_kwargs']['rope_freqs'].to(device)
            expected = payload['linear_inputs']['attn.out_proj'][:bounds[1]].to(device)
            raw = F.linear(x, w.to(device)).view(22400, 3, 56, 128)
            q = _apply_rope(F.rms_norm(raw[:, 0], (128,), q_weight.to(device), eps=1e-5), rope)
            k = _apply_rope(F.rms_norm(raw[:, 1], (128,), k_weight.to(device), eps=1e-5), rope)
            v = raw[:, 2]
        n = bounds[1]
        total = bounds[-1]
        Q, K, V = [z[:n].transpose(0, 1).unsqueeze(0).contiguous() for z in (q, k, v)]
        Qp, Kp, Vp = [z[n:].transpose(0, 1).unsqueeze(0).contiguous() for z in (q, k, v)]
        del q, k, v
        if not a.capture:
            del raw, x, w, rope
        torch.cuda.synchronize()

        def bf16():
            return F.scaled_dot_product_attention(Q, K, V, scale=128 ** -0.5)

        def sage3():
            # Official Sage3 centers K in place; real H3 routing clones K.
            return sageattn3_blackwell(Q, K.clone(), V, is_causal=False, per_block_mean=True)

        ref = bf16()
        cached_nmse = (None if expected is None else
                       metrics(ref, expected.view(n, 56, 128).transpose(0, 1).unsqueeze(0)))
        if cached_nmse is not None and cached_nmse['nmse'] > 1e-4:
            raise RuntimeError(f'Reconstructed H3 QKV did not replay cached attention: {cached_nmse}')
        arms = {'bf16_sdpa': bf16, 'sage3_official': sage3}
        source_paths = dict(sage3_api=str(Path(sage3_api.__file__).resolve()),
                            sage3_extension=str(Path(sage3_api.fp4attn_cuda.__file__).resolve()))
        if a.sage2_target is not None:
            from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda
            import sageattention
            from sageattention import core as sage2_core
            if not Path(sageattention.__file__).resolve().is_relative_to(a.sage2_target.resolve()):
                raise RuntimeError(f'SageAttention imported from unexpected environment: {sageattention.__file__}')
            if 'fp32+fp16' not in Path(sage2_core.__file__).read_text():
                raise RuntimeError('Loaded sageattention is not SageAttention 2++')
            source_paths['sage2_core'] = str(Path(sage2_core.__file__).resolve())
            source_paths['sage2_package'] = str(Path(sageattention.__file__).resolve())

            def sage2():
                # The official SM120 dispatch selects INT8 QK / FP8 PV and
                # fp32+fp16 accumulation. Its K centering is also in-place.
                return sageattn(Q, K.clone(), V, tensor_layout='HND',
                                is_causal=False, sm_scale=128 ** -0.5)

            arms['sage2pp_official'] = sage2

            def sage2_fp32():
                # Official pre-2++ FP32-buffered PV path. Keep SM120's
                # supported per-warp INT8 QK granularity fixed so only the
                # accumulation path differs from the 2++ dispatcher.
                return sageattn_qk_int8_pv_fp8_cuda(
                    Q, K.clone(), V, tensor_layout='HND', is_causal=False,
                    qk_quant_gran='per_warp', sm_scale=128 ** -0.5,
                    pv_accum_dtype='fp32+fp32')

            arms['sage2_fp32_official'] = sage2_fp32

        if a.full_helper:
            def assemble(main, padding):
                out = torch.empty((total, 56, 128), device=device, dtype=torch.bfloat16)
                out[:n] = main[0].transpose(0, 1)
                out[n:] = padding[0].transpose(0, 1)
                return out.transpose(0, 1).unsqueeze(0)

            def bf16_full():
                return assemble(bf16(), F.scaled_dot_product_attention(Qp, Kp, Vp,
                                                                      scale=128 ** -0.5))

            def sage3_full():
                return assemble(sage3(), F.scaled_dot_product_attention(Qp, Kp, Vp,
                                                                       scale=128 ** -0.5))

            arms['bf16_full_helper'] = bf16_full
            arms['sage3_full_helper'] = sage3_full
            if a.sage2_target is not None:
                def sage2_full():
                    # H3's Sage2 wrapper nudges V on both variable-length
                    # segments to avoid an exact-zero scale in padding.
                    main = sageattn(Q, K.clone(), V + 1e-8, tensor_layout='HND',
                                    is_causal=False, sm_scale=128 ** -0.5)
                    padding = sageattn(Qp, Kp.clone(), Vp + 1e-8, tensor_layout='HND',
                                       is_causal=False, sm_scale=128 ** -0.5)
                    return assemble(main, padding)

                arms['sage2pp_full_helper_guarded'] = sage2_full

                def sage2_fp32_full():
                    main = sageattn_qk_int8_pv_fp8_cuda(
                        Q, K.clone(), V + 1e-8, tensor_layout='HND',
                        is_causal=False, qk_quant_gran='per_warp',
                        sm_scale=128 ** -0.5, pv_accum_dtype='fp32+fp32')
                    padding = sageattn_qk_int8_pv_fp8_cuda(
                        Qp, Kp.clone(), Vp + 1e-8, tensor_layout='HND',
                        is_causal=False, qk_quant_gran='per_warp',
                        sm_scale=128 ** -0.5, pv_accum_dtype='fp32+fp32')
                    return assemble(main, padding)

                arms['sage2_fp32_full_helper_guarded'] = sage2_fp32_full

        report = dict(format='h3-sage-real-operator-v1',
                      timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      gpu=a.gpu, gpu_name=torch.cuda.get_device_name(device),
                      torch_version=torch.__version__, sample=str(sample_path),
                      sample_sha256=sha256(sample_path),
                      weight=None if a.capture else str(WEIGHT),
                      block=25, prompt_id=payload.get('prompt_id', 1), step=0,
                      geometry=dict(all_rows=total, valid_rows=n, padding_rows=total-n,
                                    video_tokens=payload.get('video_tokens', 21312),
                                    height=payload.get('height'), width=payload.get('width'),
                                    frames=payload.get('frames'), fps=payload.get('fps'),
                                    heads=56, head_dim=128),
                      cached_bf16_replay=cached_nmse, source_paths=source_paths,
                      warmup=a.warmup, repeats=a.repeats, batches=a.batches,
                      results={})
        for name, fn in arms.items():
            print(f'START {name}', flush=True)
            y = fn()
            reference = bf16_full() if 'full_helper' in name else ref
            row = dict(error=metrics(y, reference), timing=timed(fn, a.warmup, a.repeats, a.batches))
            if a.profile and not a.full_helper:
                row['profile_kernels'] = profile_kernels(fn, 3)
            report['results'][name] = row
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'END {name}: {row["timing"]["median_ms"]:.3f} ms, CV '
                  f'{row["timing"]["cv_percent"]:.2f}%, NMSE {row["error"]["nmse"]:.6g}',
                  flush=True)
    print(json.dumps({name: row['timing']['median_ms'] for name, row in report['results'].items()},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
