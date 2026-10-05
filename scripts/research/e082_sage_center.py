"""E082 private Sage3 center128 verification path, not a performance claim.

Only D128/BF16/noncausal/blockmean is compiled. Official Sage3 source and its
Q/K/V quantizers stay untouched. The private kernel sums its actual prequant
P values and adds mass times BF16 V mean before its original final BF16 cast.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

SOURCE = Path(__file__).with_name('e082_sage_center_src')
OFFICIAL = Path('/data1/models/svdquant-wjq/third_party/SageAttention-official-20261004/sageattention3_blackwell')
CACHE = Path('/data1/models/svdquant-wjq/research/20261005/E082')
CUDA = Path('/data1/models/svdquant-wjq/research/toolchains/cuda-12.8.1')
CUTLASS = OFFICIAL / 'csrc/cutlass'
CUDA_FLAGS = [
    '-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_OPERATORS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', '-U__CUDA_NO_BFLOAT162_OPERATORS__',
    '-U__CUDA_NO_BFLOAT162_CONVERSIONS__', '--expt-relaxed-constexpr',
    '--expt-extended-lambda', '--use_fast_math',
    '--ptxas-options=--verbose,--warn-on-local-memory-usage', '-lineinfo',
    '-DCUTLASS_DEBUG_TRACE_LEVEL=0', '-DNDEBUG', '-DQBLKSIZE=128',
    '-DKBLKSIZE=128', '-DCTA256', '-DDQINRMEM', '-DEXECMODE=0',
    '-gencode', 'arch=compute_120a,code=sm_120a', '--threads', '4',
]


def _record(path):
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def contract():
    """Read source/available binary hashes without loading extensions or CUDA."""
    files = sorted(p for p in SOURCE.iterdir() if p.suffix in {'.h', '.cu'})
    sources = {p.name: _record(p) for p in files}
    identity = hashlib.sha256(json.dumps(
        {'sources': {k: v['sha256'] for k, v in sources.items()},
         'cuda_flags': CUDA_FLAGS}, sort_keys=True).encode()).hexdigest()
    name = 'e082_sage_center128_' + identity[:16]
    directory = CACHE / ('build_' + identity[:16])
    binary = directory / (name + '.so')
    return {'name': name, 'source_identity': identity,
            'loader': _record(Path(__file__)), 'sources': sources,
            'official_sources': {p.name: _record(OFFICIAL / 'sageattn3/blackwell' / p.name)
                                 for p in files},
            'build_directory': str(directory), 'cuda_flags': CUDA_FLAGS,
            'binary': _record(binary) if binary.exists() else None,
            'recipe': 'D128 BF16 noncausal blockmean; center128 prequant P mass; FP32 correction before final cast',
            'mean': 'FP32 group mean rounded to BF16; complete video groups only',
            'residual': 'BF16(V.float() - repeated BF16 mean.float())'}


@functools.cache
def load_extension():
    """Build/load the isolated extension; explicit SM120a avoids GPU probing."""
    from torch.utils import cpp_extension
    metadata = contract()
    directory = Path(metadata['build_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    nvidia = Path(torch.__file__).resolve().parent.parent / 'nvidia'
    include_paths = [SOURCE, CUTLASS / 'include', CUTLASS / 'tools/util/include']
    include_paths += [nvidia / item / 'include' for item in ('cublas', 'cusolver', 'cusparse')]
    if not all(p.is_dir() for p in include_paths):
        raise RuntimeError(f'Missing private build include path: {include_paths}')
    previous_cuda = cpp_extension.CUDA_HOME
    cpp_extension.CUDA_HOME = str(CUDA)
    try:
        module = cpp_extension.load(
            name=metadata['name'], sources=[str(SOURCE / 'api.cu')],
            extra_cflags=['-O3', '-std=c++17'], extra_cuda_cflags=CUDA_FLAGS,
            extra_include_paths=[str(p) for p in include_paths],
            build_directory=str(directory), with_cuda=True, verbose=True)
    finally:
        cpp_extension.CUDA_HOME = previous_cuda
    (directory / 'contract.json').write_text(json.dumps(contract(), indent=2) + '\n')
    return module


@functools.cache
def _sage():
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))
    import sageattn3.api as sage
    return sage


@torch.inference_mode()
def sageattn3_center128(q, k, v, video_start, center=True):
    """Original Sage3 layout [B,H,n,128]; K is modified as in its original API.

    center=False leaves V intact and passes zero means for native byte-parity
    control. Caller owns K storage and must clone it if it will be reused.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError('E082 expects matching [B,H,n,128] Q/K/V')
    if any(x.dtype != torch.bfloat16 or not x.is_cuda for x in (q, k, v)):
        raise ValueError('E082 expects BF16 CUDA Q/K/V')
    n = q.shape[-2]
    if not 0 <= video_start <= n:
        raise ValueError('video_start must lie within the valid sequence')
    sage = _sage()
    qp, kp, vp, correction = sage.preprocess_qkv(q, k, v, per_block_mean=True)
    batch, heads, padded, dim = vp.shape
    if center:
        means = vp.float().reshape(batch, heads, padded // 128, 128, dim).mean(-2).to(torch.bfloat16)
        starts = torch.arange(padded // 128, device=vp.device) * 128
        eligible = (starts >= video_start) & (starts + 128 <= n)
        means = (means * eligible[None, None, :, None]).contiguous()
        residual = (vp.float() - means.float().repeat_interleave(128, dim=-2)).to(torch.bfloat16)
    else:
        means = torch.zeros((batch, heads, padded // 128, dim), device=vp.device, dtype=torch.bfloat16)
        residual = vp
    qlist = sage.scale_and_quant_fp4(qp)
    klist = sage.scale_and_quant_fp4_permute(kp)
    vlist = sage.scale_and_quant_fp4_transpose(residual)
    result = load_extension().fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1],
                                  correction, means, n, None, 128 ** -.5, False, True, True)
    return result[0][:, :, :n, :].contiguous()


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != '--build':
        raise SystemExit('Usage: e082_sage_center.py --build')
    load_extension()
    print(json.dumps({'status': 'built', 'cuda_initialized': torch.cuda.is_initialized(),
                      'contract': contract()}, indent=2))
