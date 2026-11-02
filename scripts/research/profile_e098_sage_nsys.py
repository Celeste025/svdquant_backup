#!/usr/bin/env python3
"""Short, labeled official-API trace for real H3 Sage2++ and Sage3 calls."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


CAPTURE = Path('/data1/models/svdquant-wjq/research/20261014/E098/'
               'h3_p1_s0_b25_480x832_73f_qkv.pt')
SAGE2 = Path('/data1/models/svdquant-wjq/research/envs/sageattention2pp-sm120-20261014')
SAGE3 = Path('/data1/models/svdquant-wjq/research/20261014/E098/sage3_sm110_probe')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError('Choose 1-10 profiled repetitions')
    sys.path[:0] = [str(SAGE2), str(SAGE3)]
    from sageattention import sageattn
    from sageattn3 import sageattn3_blackwell

    torch.cuda.set_device(args.gpu)
    payload = torch.load(CAPTURE, map_location='cpu', weights_only=True, mmap=True)
    assert payload['cu_seqlens'].tolist() == [0, 9482, 9536]
    q, k, v = [payload[name][:9482].to(f'cuda:{args.gpu}').transpose(0, 1).unsqueeze(0)
               for name in ('q', 'k', 'v')]
    assert all(t.shape == (1, 56, 9482, 128) and not t.is_contiguous() for t in (q, k, v))
    k_work = torch.empty_like(k)

    def sage2pp() -> torch.Tensor:
        return sageattn(q, k, v, tensor_layout='HND', is_causal=False,
                        sm_scale=128 ** -0.5)

    def sage3() -> torch.Tensor:
        # API centers K in place; this is a fresh disposable K from H3's QKV.
        k_work.copy_(k)
        return sageattn3_blackwell(q, k_work, v, is_causal=False,
                                   per_block_mean=True)

    with torch.inference_mode():
        for fn in (sage2pp, sage3):
            for _ in range(2):
                fn()
        torch.cuda.synchronize()
        if torch.cuda.cudart().cudaProfilerStart() != 0:
            raise RuntimeError('cudaProfilerStart failed')
        for name, fn in (('sage2pp_official_api', sage2pp), ('sage3_official_api', sage3)):
            for repeat in range(args.repeats):
                torch.cuda.nvtx.range_push(f'e098_{name}_repeat_{repeat}')
                fn()
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if torch.cuda.cudart().cudaProfilerStop() != 0:
            raise RuntimeError('cudaProfilerStop failed')
    print(f'Captured {args.repeats} public-API calls per arm on GPU {args.gpu}', flush=True)


if __name__ == '__main__':
    main()
