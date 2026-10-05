#!/usr/bin/env python3
"""Packing correctness only: frozen E005 bytes plus historical QDQ values.

Reference functions are compiled unchanged from their AST nodes to avoid
importing/model-initializing the historical pipeline. Source file SHA256s are
recorded. There is no performance claim or GEMM/model benchmark here.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
from h3_nvfp4_fastpack import (collect_fastpack_checks, pack_activation_fast,
                               pack_legacy_h3, validate_flags)
from wan_native_nvfp4 import swizzle_scales, unswizzle_scales

ROOT = Path(__file__).resolve().parents[2]


def frozen_references():
    env = {'torch': torch}
    specifications = [
        (ROOT / 'scripts/research/probe_h3_native_contract.py', {'quantize_pack', 'decode'}),
        (ROOT / 'scripts/minimax_h3_svdquant_common.py', {'_fp8_round_positive', 'nvfp4_qdq', 'FP4_VALUES'}),
    ]
    for path, names in specifications:
        nodes = []
        for node in ast.parse(path.read_text()).body:
            name = getattr(node, 'name', None)
            if isinstance(node, ast.Assign):
                name = getattr(node.targets[0], 'id', None)
            if name in names:
                nodes.append(node)
        if len(nodes) != len(names):
            raise RuntimeError(f'Frozen reference AST selection incomplete: {path}')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def byte_diff(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(f'Byte comparison metadata mismatch: {a.shape}/{a.dtype}, {b.shape}/{b.dtype}')
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def synthetic_cases():
    cases = []
    # Force g=1 and each nonsentinel group SF=1. Preserve both true signed
    # zeros and negative nonzero values which round to zero.
    values = [.25, .75, 1.25, 1.75, 2.5, 3.5, 5., -.25, -.75, -1.25,
              -1.75, -2.5, -3.5, -5., 6., -6., 0., -0., .125, -.125,
              .2490234375, -.2490234375, .251953125, -.251953125,
              .5, -.5, 1., -1., 2., -2., 6., -6.]
    ties = torch.tensor(values, dtype=torch.bfloat16).repeat(17, 1)
    ties[-1] = 2688
    cases.append(('e2m1_signed_midpoints_and_zero', lambda: ties.cuda()))
    fp8 = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    # The zero/min-subnormal midpoint rounds to SF0 with nonzero input and
    # belongs to the explicitly rejected domain tested below.
    sf_ties = (((fp8[1:-1] + fp8[2:])/2)[:, None]*6).repeat(1, 32).bfloat16()
    sf_ties = torch.cat((sf_ties, torch.full((1, 32), 2688, dtype=torch.bfloat16)))
    cases.append(('e4m3_midpoints_including_subnormals', lambda: sf_ties.cuda()))
    cases.append(('all_zero_with_negative_zero', lambda: torch.tensor([0., -0.], dtype=torch.bfloat16,
                  device='cuda').repeat(129, 32)))
    mixed = torch.zeros((129, 48), dtype=torch.bfloat16)
    mixed[-1] = 2688
    mixed[0, :16] = .01171875
    cases.append(('zero_sf_zero_groups_and_subnormal_sf_tail', lambda: mixed.cuda()))
    tiny = torch.tensor([2**-126, -(2**-126), 2**-133, -(2**-133)], dtype=torch.bfloat16).repeat(7, 8)
    cases.append(('bf16_small_normal_and_subnormal', lambda: tiny.cuda()))
    cases.append(('tail_257_48', lambda: torch.randn((257, 48), device='cuda', dtype=torch.bfloat16)))
    cases.append(('batched_shape_3_17_32', lambda: torch.randn((3, 17, 32), device='cuda', dtype=torch.bfloat16)))
    return cases


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    ref = frozen_references()
    sources = [Path(__file__), Path(__file__).with_name('h3_nvfp4_fastpack.py'),
               ROOT/'scripts/research/probe_h3_native_contract.py',
               ROOT/'scripts/minimax_h3_svdquant_common.py', Path(__file__).with_name('wan_native_nvfp4.py')]
    report = dict(status='running', started=time.time(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  scope='packing correctness; no performance measurement', seed=20261002,
                  reference='unchanged function ASTs from frozen E005 and H3 common',
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  cases=[], rejections={})
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    cases = [] if args.real_only else synthetic_cases()
    if not args.skip_real_shapes and not args.real_only:
        for suffix, k in [('qkv', 5376), ('out', 7168), ('fc1', 5376), ('fc2', 14336)]:
            cases.append((f'random_realshape_{suffix}_22400_{k}',
                          lambda k=k: torch.randn((22400, k), device='cuda', dtype=torch.bfloat16)))
    if args.inputs:
        # Loading real captures is deferred until their cases to limit RAM/GPU
        # residency. Combined audit captures and systems' per-layer captures
        # both use an explicit x_s field (or a top-level tensor).
        for path in args.inputs:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if 'inputs' in payload:
                entries = payload['inputs']
            elif 'x_s' in payload:
                entries = {path.stem: payload['x_s']}
            else:
                entries = payload
            for name, value in entries.items():
                if isinstance(value, dict):
                    value = value.get('x_s', value.get('xs', value.get('input')))
                if torch.is_tensor(value):
                    cases.append((f'real_{path.stem}_{name}', lambda value=value: value.cuda()))
            report.setdefault('input_files', []).append({'path': str(path), 'bytes': path.stat().st_size,
                                                        'mtime_ns': path.stat().st_mtime_ns})
    if args.real_only and not cases:
        raise ValueError('--real-only requires real tensor inputs')
    try:
        for name, factory in cases:
            x = factory().contiguous()
            before = time.time()
            flat = x.reshape(-1, x.shape[-1])
            expected = ref['quantize_pack'](flat, args.chunk_rows)
            got = pack_legacy_h3(x)
            expected_swizzled = swizzle_scales(expected['scale'])
            torch.cuda.synchronize()
            row = dict(name=name, shape=list(x.shape), domain_flags=got.domain_flags.tolist(),
                       byte_mismatches=dict(codes=byte_diff(got.codes, expected['legacy']),
                                            global_scale=byte_diff(got.global_scale, expected['global']),
                                            swizzled_scales_including_padding=byte_diff(got.scales, expected_swizzled)),
                       reference_stats=expected['stats'])
            report['cases'].append(row)
            save()
            validate_flags([got.domain_flags])
            if any(row['byte_mismatches'].values()):
                row['debug_global'] = [float(got.global_scale), float(expected['global'])]
                mismatch = (got.codes != expected['legacy']).nonzero()[:10]
                row['debug_codes'] = [dict(position=p.tolist(), got=int(got.codes[tuple(p)]),
                                           reference=int(expected['legacy'][tuple(p)])) for p in mismatch]
                got_sf = unswizzle_scales(got.scales, flat.shape[0], flat.shape[1]//16)
                sf_mismatch = (got_sf.view(torch.uint8) != expected['scale'].view(torch.uint8)).nonzero()[:10]
                row['debug_scales'] = [dict(position=p.tolist(), got=float(got_sf[tuple(p)]),
                                            reference=float(expected['scale'][tuple(p)])) for p in sf_mismatch]
                save()
                raise AssertionError(f'E005 packing byte mismatch: {name}')
            decoded = ref['decode'](got.codes, expected['scale'], got.global_scale,
                                    dtype=torch.bfloat16, chunk=args.chunk_rows).reshape(x.shape)
            qdq = ref['nvfp4_qdq'](x, element_size=args.chunk_rows)
            row['decoded_equal_historical_qdq'] = torch.equal(decoded, qdq)
            row['decoded_vs_qdq_byte_differences'] = byte_diff(decoded, qdq)
            row['decode_zero_sign_note'] = ('E005 keeps a negative-zero nibble for small negative z; '
                                           'historical signed15 QDQ uses +0. Equal values are required; '
                                           'the packed bytes themselves match E005 exactly.')
            if not row['decoded_equal_historical_qdq']:
                save()
                raise AssertionError(f'Historical QDQ value mismatch: {name}')
            if x.shape[-1] % 32 == 0:
                packet = pack_activation_fast(x, chunk_rows=args.chunk_rows)
                assert packet.original_shape == tuple(x.shape)
                assert byte_diff(packet.packed, expected['legacy']) == 0
                del packet
            row['seconds_not_benchmark'] = time.time()-before
            print(json.dumps(row), flush=True)
            save()
            del expected, expected_swizzled, got, decoded, qdq, flat, x
            gc.collect()
        if not args.real_only:
            underflow = torch.full((129, 32), 1e-8, device='cuda', dtype=torch.bfloat16)
            underflow[-1] = 2688
            flags = pack_legacy_h3(underflow).domain_flags.tolist()
            assert flags[1] == 1, flags
            report['nonzero_sf0_flags'] = flags
            invalid_cases = [('nonzero_sf0', underflow)]
            midpoint_underflow = underflow.clone()
            midpoint_underflow[:-1] = 6*(2**-10)
            invalid_cases.append(('e4m3_zero_min_subnormal_midpoint', midpoint_underflow))
            for label, value in [('nan', float('nan')), ('inf', float('inf')), ('negative_inf', -float('inf'))]:
                bad = torch.ones((129, 64), device='cuda', dtype=torch.bfloat16)
                bad[7, 17] = value
                invalid_cases.append((label, bad))
            for name, bad in invalid_cases:
                try:
                    pack_activation_fast(bad)
                except ValueError:
                    report['rejections'][name] = True
                else:
                    raise AssertionError(f'Unsafe input not rejected: {name}')
            try:
                with collect_fastpack_checks() as checks:
                    pack_activation_fast(torch.zeros_like(underflow))
                    pack_activation_fast(underflow)
                    assert len(checks) == 2
            except ValueError:
                report['rejections']['deferred_check_after_valid_then_invalid'] = True
            else:
                raise AssertionError('Deferred flags were not checked')
            # The context must restore the default immediate check after error.
            try:
                pack_activation_fast(underflow)
            except ValueError:
                report['rejections']['context_restoration'] = True
            else:
                raise AssertionError('Immediate validation was lost')
        report.update(status='complete', elapsed_seconds_not_benchmark=time.time()-report['started'])
        save()
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E009_fastpack_parity.json')
    parser.add_argument('--inputs', type=Path, nargs='+')
    parser.add_argument('--skip-real-shapes', action='store_true')
    parser.add_argument('--real-only', action='store_true')
    parser.add_argument('--chunk-rows', type=int, default=128)
    run(parser.parse_args())
