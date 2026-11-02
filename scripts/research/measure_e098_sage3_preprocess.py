#!/usr/bin/env python3
"""Stage-level CUDA-event timing of the official Sage3 H3 online call."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F

from bench_h3_sage_real_gpu0 import metrics


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE3 = Path('/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe')
OUTPUT = Path('results/research/E098/sage3_sm110_preprocess_stages_gpu0.json')
STAGES = ('center_pad_qkv_and_delta_s', 'quantize_q_fp4', 'quantize_k_fp4',
          'quantize_v_fp4', 'fp4_attention_core', 'crop_output')
PREPROCESS_STAGES = ('mean_k', 'subtract_k_mean', 'pad_q', 'pad_k', 'pad_v',
                     'group_mean_q', 'delta_s_matmul_bf16', 'delta_s_cast_fp32')
WARMUP, BATCHES, REPEATS = 25, 12, 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sage3-dir', type=Path, default=SAGE3)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sage3_dir))
    from sageattn3 import sageattn3_blackwell
    from sageattn3.api import (blockscaled_fp4_attn, preprocess_qkv,
                               scale_and_quant_fp4, scale_and_quant_fp4_permute,
                               scale_and_quant_fp4_transpose, triton_group_mean)

    torch.cuda.set_device(0)
    payload = torch.load(args.capture, map_location='cpu', weights_only=True, mmap=True)
    if payload['q'].ndim == 3:
        assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
        n = 9482
        q, k, v = [payload[name][:n].to('cuda:0').transpose(0, 1).unsqueeze(0)
                   for name in ('q', 'k', 'v')]
        input_policy = 'H3 noncontiguous HND; K restored before first event'
    elif payload['q'].ndim == 4:
        q, k, v = [payload[name].to('cuda:0') for name in ('q', 'k', 'v')]
        n = q.shape[2]
        input_policy = 'H3 contiguous HND; K restored before first event'
    else:
        raise RuntimeError('unsupported QKV capture rank')
    assert all(z.shape == (1, 56, n, 128) for z in (q, k, v))
    n_pad = (-n) % 128
    k_work = torch.empty_like(k)

    def one_call(record):
        # As in the main two-scope benchmark: restore disposable K before the
        # first start event, excluding only this benchmark-specific copy.
        k_work.copy_(k)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)] if record else None
        if record:
            events[0].record()
        qp, kp, vp, delta_s = preprocess_qkv(q, k_work, v, True)
        if record:
            events[1].record()
        qlist = scale_and_quant_fp4(qp)
        if record:
            events[2].record()
        klist = scale_and_quant_fp4_permute(kp)
        if record:
            events[3].record()
        vlist = scale_and_quant_fp4_transpose(vp)
        if record:
            events[4].record()
        out = blockscaled_fp4_attn(qlist, klist, vlist, delta_s, n,
                                   is_causal=False, per_block_mean=True,
                                   is_bf16=True)[0]
        if record:
            events[5].record()
        out = out[:, :, :n, :].contiguous()
        if record:
            events[6].record()
        return out, events, (qp, kp, vp, delta_s, qlist, klist, vlist)

    def preprocess_detail(record):
        k_work.copy_(k)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(9)] if record else None
        if record:
            events[0].record()
        km = k_work.mean(dim=-2, keepdim=True)
        if record:
            events[1].record()
        k_work.sub_(km)
        if record:
            events[2].record()
        qp = F.pad(q, (0, 0, 0, n_pad), value=0).contiguous()
        if record:
            events[3].record()
        kp = F.pad(k_work, (0, 0, 0, n_pad), value=0).contiguous()
        if record:
            events[4].record()
        vp = F.pad(v, (0, 0, 0, n_pad), value=0).contiguous()
        if record:
            events[5].record()
        qp, qm = triton_group_mean(qp)
        if record:
            events[6].record()
        delta_s_bf16 = torch.matmul(qm, kp.transpose(-2, -1))
        if record:
            events[7].record()
        delta_s = delta_s_bf16.to(torch.float32).contiguous()
        if record:
            events[8].record()
        return (qp, kp, vp, delta_s), events

    with torch.inference_mode():
        k_ref = k.clone()
        official = sageattn3_blackwell(q, k_ref, v, is_causal=False,
                                       per_block_mean=True)
        replay, _, tensors = one_call(False)
        agreement = metrics(replay, official)
        if agreement['nmse'] > 1e-8:
            raise RuntimeError(f'Staged official functions disagree with API: {agreement}')
        qp, kp, vp, delta_s, qlist, klist, vlist = tensors
        detail_tensors, _ = preprocess_detail(False)
        if not all(torch.equal(left, right) for left, right in
                   zip(detail_tensors, (qp, kp, vp, delta_s))):
            raise RuntimeError('Detailed preprocess does not reproduce official preprocess')
        shapes = {name: list(t.shape) for name, t in (
            ('q_padded', qp), ('k_padded', kp), ('v_padded', vp),
            ('delta_s', delta_s), ('q_codes', qlist[0]),
            ('q_scales', qlist[1]), ('k_codes', klist[0]),
            ('k_scales', klist[1]), ('v_codes', vlist[0]),
            ('v_scales', vlist[1]))}
        for _ in range(WARMUP):
            one_call(False)
            preprocess_detail(False)
        torch.cuda.synchronize()
        batch_means = []
        detail_batch_means = []
        for _ in range(BATCHES):
            all_events = []
            detail_events = []
            for _ in range(REPEATS):
                _, events, _ = one_call(True)
                all_events.append(events)
                _, events = preprocess_detail(True)
                detail_events.append(events)
            torch.cuda.synchronize()
            batch_means.append([statistics.mean(events[i].elapsed_time(events[i + 1])
                                                for events in all_events)
                                for i in range(len(STAGES))])
            detail_batch_means.append([statistics.mean(events[i].elapsed_time(events[i + 1])
                                                       for events in detail_events)
                                       for i in range(len(PREPROCESS_STAGES))])
        rows = {}
        for i, name in enumerate(STAGES):
            values = [batch[i] for batch in batch_means]
            mean_ms = statistics.mean(values)
            rows[name] = dict(mean_ms=mean_ms,
                              cv_batch_mean_percent=100 * statistics.stdev(values) / mean_ms,
                              batch_mean_ms=values)
            print(f'{name}: {mean_ms:.4f} ms, CV {rows[name]["cv_batch_mean_percent"]:.2f}%',
                  flush=True)
        pack_ms = sum(rows[name]['mean_ms'] for name in STAGES[1:4])
        noncore_ms = sum(rows[name]['mean_ms'] for name in STAGES[:4]) + rows['crop_output']['mean_ms']
        print(f'FP4 QKV pack total: {pack_ms:.4f} ms; all non-core: {noncore_ms:.4f} ms',
              flush=True)
        detail_rows = {}
        for i, name in enumerate(PREPROCESS_STAGES):
            values = [batch[i] for batch in detail_batch_means]
            mean_ms = statistics.mean(values)
            detail_rows[name] = dict(mean_ms=mean_ms,
                                     cv_batch_mean_percent=100 * statistics.stdev(values) / mean_ms,
                                     batch_mean_ms=values)
            print(f'{name}: {mean_ms:.4f} ms, CV {detail_rows[name]["cv_batch_mean_percent"]:.2f}%',
                  flush=True)

    report = dict(format='e098-sage3-preprocess-stages-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  sage3_source=str(args.sage3_dir), capture=str(args.capture),
                  shape_hnd=[1, 56, n, 128], official_api_nmse=agreement['nmse'],
                  warmup=WARMUP, batches=BATCHES, repeats=REPEATS,
                  input_policy=input_policy,
                  tensor_shapes=shapes, stages=rows,
                  preprocess_detail_stages=detail_rows,
                  fp4_qkv_pack_total_ms=pack_ms, noncore_total_ms=noncore_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
