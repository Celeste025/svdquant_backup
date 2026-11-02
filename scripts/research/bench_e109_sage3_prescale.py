#!/usr/bin/env python3
"""Matched real-H3 Sage3 benchmark for official SM110 and prescale builds.

Run each build in a separate process: both expose extensions named fp4attn_cuda.
The timing boundary matches E098's two-scope benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from bench_h3_sage_real_gpu0 import metrics
from bench_h3_sage_two_scope import measure


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sage3-dir', type=Path, required=True)
    parser.add_argument('--module', choices=('sageattn3', 'sageattn3_prescale'), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    source = args.sage3_dir.resolve()
    extension = source / 'fp4attn_cuda.cpython-312-x86_64-linux-gnu.so'
    if not extension.is_file():
        raise FileNotFoundError(extension)
    sys.path.insert(0, str(source))
    api = importlib.import_module(f'{args.module}.api')
    torch.cuda.set_device(args.gpu)

    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    assert payload['format'] == 'h3-480p3s-real-attention-v1'
    assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
    n, heads, dim = 9482, 56, 128
    q, k, v = [payload[name][:n].to(f'cuda:{args.gpu}').transpose(0, 1).unsqueeze(0)
               for name in ('q', 'k', 'v')]
    assert q.shape == k.shape == v.shape == (1, heads, n, dim)
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    k_original = k.clone()
    k_work = torch.empty_like(k)
    operations = 4 * heads * n * n * dim
    rows = {}

    def record(scope, fn, ref, prepare=lambda: None, crop=lambda x: x):
        prepare()
        output = crop(fn())
        error = metrics(output, ref)
        timing = measure(fn, prepare)
        rows[scope] = dict(timing=timing,
                           tops=operations / 1e12 / (timing['mean_ms'] / 1000),
                           nmse_vs_bf16=error['nmse'])
        print(f'{scope}: {timing["mean_ms"]:.4f} ms, {rows[scope]["tops"]:.1f} TOPS, '
              f'NMSE={error["nmse"]:.8f}, CV={timing["cv_batch_mean_percent"]:.2f}%',
              flush=True)
        return output

    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, scale=dim ** -0.5)

        def online():
            return api.sageattn3_blackwell(q, k_work, v, is_causal=False,
                                           per_block_mean=True)

        online_output = record('online', online, ref, lambda: k_work.copy_(k))
        assert torch.equal(k, k_original), 'The benchmark source K was modified'

        k_pre = k.clone()
        q3, k3, v3, delta_s = api.preprocess_qkv(q, k_pre, v, True)
        qlist = api.scale_and_quant_fp4(q3)
        klist = api.scale_and_quant_fp4_permute(k3)
        vlist = api.scale_and_quant_fp4_transpose(v3)

        def core():
            return api.blockscaled_fp4_attn(qlist, klist, vlist, delta_s, n,
                                            is_causal=False, per_block_mean=True,
                                            is_bf16=True)[0]

        crop = lambda x: x[:, :, :n, :].contiguous()
        agreement = metrics(crop(core()), online_output)['nmse']
        if agreement > 1e-8:
            raise RuntimeError(f'Core and online output disagree: NMSE={agreement}')
        record('core_only', core, ref, crop=crop)
        rows['core_only']['nmse_vs_online'] = agreement

    report = dict(format='e109-h3-sage3-prescale-two-scope-v1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(source), module=args.module, gpu=args.gpu,
                  gpu_name=torch.cuda.get_device_name(args.gpu),
                  capture=str(CAPTURE), shape_hnd=[1, heads, n, dim],
                  is_causal=False, per_block_mean=True,
                  fp4attn_sha256=sha256(extension),
                  operations=operations,
                  operation_convention='4*B*H*N*N*D, MAC=2 ops',
                  timing='CUDA event mean of 12 batch means x 12 calls; 25 warmups',
                  online_k_policy='Restored disposable K before each event; copy excluded',
                  core_policy='QKV preprocess excluded; core result cropped only for accuracy',
                  results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
