#!/usr/bin/env python3
"""CUDA-event timing of Sage2++ official SM120 preprocessing on real H3 QKV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch

from bench_h3_sage_real_gpu0 import metrics


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE2 = Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')
OUTPUT = Path('results/research/E098/sage2pp_preprocess_stages_gpu0.json')
STAGES = ('mean_k', 'quantize_q_int8', 'quantize_k_int8_fused_center',
          'transpose_pad_permute_v', 'quantize_v_fp8', 'attention_core')
WARMUP, BATCHES, REPEATS = 25, 12, 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.path.insert(0, str(SAGE2))
    from sageattention import sageattn, _fused, _qattn_sm89
    from sageattention.quant import per_channel_fp8, per_warp_int8

    torch.cuda.set_device(0)
    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
    n, h, d = 9482, 56, 128
    q, k, v = [payload[name][:n].to('cuda:0').transpose(0, 1).unsqueeze(0)
               for name in ('q', 'k', 'v')]
    assert all(z.shape == (1, h, n, d) and not z.is_contiguous()
               for z in (q, k, v))
    scale = d ** -0.5
    padded_n = (n + 63) // 64 * 64

    def one_call(record):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)] if record else None
        if record:
            events[0].record()
        km = k.mean(dim=2, keepdim=True)
        if record:
            events[1].record()
        qi = torch.empty(q.shape, dtype=torch.int8, device=q.device)
        qs = torch.empty((1, h, ((n + 127) // 128) * 4), dtype=torch.float32, device=q.device)
        _fused.quant_per_warp_int8_cuda(q, qi, qs, 128, 32, 1)
        if record:
            events[2].record()
        ki = torch.empty(k.shape, dtype=torch.int8, device=k.device)
        ks = torch.empty((1, h, (n + 63) // 64), dtype=torch.float32, device=k.device)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km.squeeze(2), ki, ks, 64, 1)
        if record:
            events[3].record()
        vt = torch.empty((1, h, d, padded_n), dtype=v.dtype, device=v.device)
        _fused.transpose_pad_permute_cuda(v, vt, 1)
        if record:
            events[4].record()
        vi = torch.empty(vt.shape, dtype=torch.float8_e4m3fn, device=v.device)
        vs = torch.empty((1, h, d), dtype=torch.float32, device=v.device)
        _fused.scale_fuse_quant_cuda(vt, vi, vs, n, 2.25, 1)
        if record:
            events[5].record()
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            qi, ki, vi, out, qs, ks, vs, 1, 0, 2, scale, 0)
        if record:
            events[6].record()
        return out, events, (qi, qs, ki, ks, vi, vs)

    with torch.inference_mode():
        official = sageattn(q, k, v, tensor_layout='HND', is_causal=False,
                            sm_scale=scale)
        replay, _, tensors = one_call(False)
        agreement = metrics(replay, official)
        if agreement['nmse'] > 1e-8:
            raise RuntimeError(f'Staged Sage2++ differs from official API: {agreement}')
        km = k.mean(dim=2, keepdim=True)
        official_quant = (*per_warp_int8(q, k, km, tensor_layout='HND',
                                        BLKQ=128, WARPQ=32, BLKK=64),
                          *per_channel_fp8(v, tensor_layout='HND',
                                           scale_max=2.25, smooth_v=False)[:2])
        if not all(torch.equal(left, right) for left, right in
                   zip(tensors, official_quant)):
            raise RuntimeError('Staged Sage2++ quant tensors differ from official helpers')
        shapes = {name: list(t.shape) for name, t in
                  zip(('q_int8', 'q_scale', 'k_int8', 'k_scale', 'v_fp8', 'v_scale'), tensors)}
        for _ in range(WARMUP):
            one_call(False)
        torch.cuda.synchronize()
        batch_means = []
        for _ in range(BATCHES):
            all_events = []
            for _ in range(REPEATS):
                _, events, _ = one_call(True)
                all_events.append(events)
            torch.cuda.synchronize()
            batch_means.append([statistics.mean(events[i].elapsed_time(events[i + 1])
                                                for events in all_events)
                                for i in range(len(STAGES))])
        rows = {}
        for i, name in enumerate(STAGES):
            values = [batch[i] for batch in batch_means]
            mean_ms = statistics.mean(values)
            rows[name] = dict(mean_ms=mean_ms,
                              cv_batch_mean_percent=100 * statistics.stdev(values) / mean_ms,
                              batch_mean_ms=values)
            print(f'{name}: {mean_ms:.4f} ms, CV {rows[name]["cv_batch_mean_percent"]:.2f}%',
                  flush=True)
        noncore_ms = sum(rows[name]['mean_ms'] for name in STAGES[:-1])
        total_ms = noncore_ms + rows['attention_core']['mean_ms']
        print(f'Non-core: {noncore_ms:.4f} ms; staged total: {total_ms:.4f} ms',
              flush=True)

    report = dict(format='e098-sage2pp-preprocess-stages-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  sage2_source=str(SAGE2), capture=str(CAPTURE),
                  shape_hnd=[1, h, n, d], official_api_nmse=agreement['nmse'],
                  warmup=WARMUP, batches=BATCHES, repeats=REPEATS,
                  input_policy='H3 noncontiguous HND, no cloning',
                  tensor_shapes=shapes, stages=rows,
                  noncore_total_ms=noncore_ms, staged_total_ms=total_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
