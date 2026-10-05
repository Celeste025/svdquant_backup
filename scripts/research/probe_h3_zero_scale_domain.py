#!/usr/bin/env python3
"""E009 failure diagnostic: reproduce block1.fc2, audit SF=0 without editing packers."""
from __future__ import annotations
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from minimax_h3_svdquant_common import (H3_DIT_PATH, TARGET_SUFFIXES, load_h3_pipeline,
                                        raw_call_to_block0, tree_device, nvfp4_qdq)
from probe_h3_modality_pulse import sha256, tensor_sha, summarize, partitions, save
from probe_h3_native_contract import quantize_pack, decode
from h3_native_nvfp4 import NativeH3Linear
from wan_native_nvfp4 import PackedNVFP4, swizzle_scales

DATA = Path('/data1/models/svdquant-wjq/research/20261002/E009')


class Captured(RuntimeError):
    pass


@torch.inference_mode()
def execute(args, report):
    torch.set_num_threads(6)
    previous = json.loads(args.reference.read_text())
    if previous['status'] != 'failed_stop' or 'Nonzero H3 groups underflow' not in previous['error']:
        raise RuntimeError('Expected preserved strict-domain failure report')
    for path in [ROOT/'scripts/minimax_h3_svdquant_common.py', Path(__file__).with_name('h3_native_nvfp4.py'),
                 Path(__file__).with_name('wan_native_nvfp4.py')]:
        if sha256(path) != previous['files'][str(path)]['sha256']:
            raise RuntimeError(f'Frozen source changed since failure: {path}')
    report['files'] = {str(p): {'sha256': sha256(p)} for p in
                       [Path(__file__), args.reference, args.sample,
                        Path(__file__).with_name('probe_h3_native_contract.py')]}
    report['source_failure_sha256'] = sha256(args.reference)
    manifest_path = args.export_dir/'manifest.json'
    if sha256(manifest_path) != previous['export_manifest_sha256']:
        raise RuntimeError('Frozen export changed')
    manifest = json.loads(manifest_path.read_text())
    sample = torch.load(args.sample, map_location='cpu', weights_only=False, mmap=True)
    if sha256(args.sample) != previous['files'][str(args.sample)]['sha256']:
        raise RuntimeError('Fixed sample changed')
    attention = importlib.import_module('diffsynth.core.attention.attention')
    if attention.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('Explicit original torch attention required')
    pipe = load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    dit = pipe.dit.eval()
    names = [f'blocks.{i}.{s}' for i in (0, 1) for s in TARGET_SUFFIXES]
    report['installed_names'] = names
    for row in manifest['layers']:
        if row['name'] not in names:
            continue
        path = args.export_dir/row['file']
        if sha256(path) != row['file_sha256']:
            raise RuntimeError('Export layer SHA mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        old = dit.get_submodule(row['name'])
        replacement = NativeH3Linear.from_export(payload, device='cuda', chunk_rows=1024)
        parent, attr = row['name'].rsplit('.', 1)
        setattr(dit.get_submodule(parent), attr, replacement)
        old.weight = old.bias = None
    hidden, kw = raw_call_to_block0(dit, sample)
    hidden = hidden.cuda()
    kw = tree_device(kw, 'cuda')
    first = dit.blocks[0](hidden, **kw)
    report['block0_sha256'] = tensor_sha(first)
    report['failed_run_block0_sha256'] = previous['stages']['native']['blocks'][0]['sha256']
    if report['block0_sha256'] != report['failed_run_block0_sha256']:
        raise RuntimeError('Prefix does not reproduce original native block0')
    target = dit.blocks[1].mlp.fc2
    captured = {}
    def capture(module, inputs):
        captured['raw'] = inputs[0].detach().cpu()
        captured['x_s'] = (inputs[0]/module.smooth).detach().cpu()
        raise Captured('Stop at the original failing input')
    handle = target.register_forward_pre_hook(capture)
    try:
        try:
            dit.blocks[1](first, **kw)
        except Captured:
            pass
    finally:
        handle.remove()
    if set(captured) != {'raw', 'x_s'}:
        raise RuntimeError('Failed to capture block1.fc2')
    path = args.artifact_dir/'block1_fc2_inputs.pt'
    torch.save(captured, path)
    report['capture'] = {'file': str(path), 'sha256': sha256(path), 'layer': 'blocks.1.mlp.fc2',
                         'x_s_sha256': tensor_sha(captured['x_s']), 'shape': list(captured['x_s'].shape),
                         'trajectory': 'same native block0 SHA, actual native block1 upstream; fixed p1step0'}
    save(report, args.output)
    print(f'E009 SF0 captured {path}', flush=True)
    x = captured['x_s'].cuda()
    del hidden, kw, first, captured
    try:
        target.pack_input(x)
    except ValueError as exc:
        if 'Nonzero H3 groups underflow' not in str(exc):
            raise
        report['strict_guard_reproduced'] = str(exc)
    else:
        raise RuntimeError('Original strict guard no longer reproduced')
    packed = quantize_pack(x, chunk=1024)
    original = nvfp4_qdq(x, element_size=1024)
    decoded = decode(packed['legacy'], packed['scale'], packed['global'], dtype=torch.bfloat16, chunk=1024)
    numeric = summarize(decoded, original)
    byte_diff = decoded.view(torch.int16) != original.view(torch.int16)
    zero_sign = byte_diff & (decoded == 0) & (original == 0)
    parity = {'finite_common': bool(torch.isfinite(original).all()),
              'finite_decoded': bool(torch.isfinite(decoded).all()),
              'torch_equal': torch.equal(decoded, original), 'metrics': numeric,
              'zero_sign_element_differences': int(zero_sign.sum()),
              'other_byte_differences': int((byte_diff & ~zero_sign).sum())}
    report['legacy_qdq_vs_e005_decode'] = parity
    report['e005_pack_stats'] = packed['stats']
    save(report, args.output)
    if not all(parity[k] for k in ('finite_common', 'finite_decoded', 'torch_equal')) or parity['other_byte_differences']:
        raise RuntimeError('E005 zero-scale decode does not match old QDQ')
    idx, description = partitions(sample, x.shape[0])
    idx['valid'] = torch.cat([idx[k] for k in ('video', 'audio', 'text')]).sort().values
    idx['all'] = torch.arange(x.shape[0])
    report['partitions'] = description
    row_stats = {key: torch.zeros(x.shape[0], dtype=dtype) for key, dtype in
                 [('groups_sf0', torch.int64), ('groups_nonzero_sf0', torch.int64),
                  ('nonzero_values_sf0', torch.int64), ('input_energy', torch.float64),
                  ('sf0_input_energy', torch.float64), ('sf0_input_absmax', torch.float64),
                  ('nonzero_common_outputs_sf0', torch.int64)]}
    for begin in range(0, x.shape[0], 1024):
        stop = min(begin+1024, x.shape[0])
        values = x[begin:stop].float().reshape(stop-begin, -1, 16)
        sf0 = packed['scale'][begin:stop].float() == 0
        maximum = values.abs().amax(-1)
        energy = values.square().sum(-1, dtype=torch.float64)
        row_stats['groups_sf0'][begin:stop] = sf0.sum(-1).cpu()
        row_stats['groups_nonzero_sf0'][begin:stop] = (sf0 & (maximum != 0)).sum(-1).cpu()
        row_stats['nonzero_values_sf0'][begin:stop] = ((values != 0) & sf0.unsqueeze(-1)).sum((1, 2)).cpu()
        row_stats['input_energy'][begin:stop] = energy.sum(-1).cpu()
        row_stats['sf0_input_energy'][begin:stop] = (energy*sf0).sum(-1).cpu()
        row_stats['sf0_input_absmax'][begin:stop] = (maximum*sf0).amax(-1).cpu()
        qvalues = original[begin:stop].reshape(stop-begin, -1, 16)
        row_stats['nonzero_common_outputs_sf0'][begin:stop] = ((qvalues != 0) & sf0.unsqueeze(-1)).sum((1, 2)).cpu()
    report['zero_scale_by_modality'] = {}
    for name, indices in idx.items():
        row = {key: (float(values[indices].max()) if key == 'sf0_input_absmax' else float(values[indices].sum()))
               for key, values in row_stats.items()}
        row['tokens'] = indices.numel()
        row['groups'] = indices.numel()*(x.shape[1]//16)
        row['sf0_input_energy_fraction'] = row['sf0_input_energy']/max(row['input_energy'], 1e-30)
        report['zero_scale_by_modality'][name] = row
    packet = PackedNVFP4(packed['legacy'], packed['scale'], packed['global'],
                         swizzle_scales(packed['scale']), tuple(x.shape), 'E005_legacy_canonical_zero_sf0')
    ref = target.main_from_packet(packet, mode='packed_qdq', include_bias=False)
    got = target.main_from_packet(packet, mode='native', include_bias=False)
    report['samepacket_full_main'] = {'all': summarize(got, ref),
                                     **{name: summarize(got, ref, indices.cuda()) for name, indices in idx.items() if name != 'all'}}
    report['samepacket_full_main']['finite'] = bool(torch.isfinite(got).all())
    if not report['samepacket_full_main']['finite'] or report['samepacket_full_main']['all']['nmse'] > 1e-4:
        raise RuntimeError('Actual native GEMM with canonical SF0 failed samepacket main gate')
    report['status'] = 'complete'
    report['interpretation'] = ('For this exact failing activation, E005 canonical-zero SF0 decodes numerically exactly as old common QDQ; '
                                'zero-scale lost energy belongs to the historical quantizer too. This does not establish a full-model native baseline.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'results/research/E009_h3_full.json')
    parser.add_argument('--sample', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt')
    parser.add_argument('--export-dir', type=Path, default=DATA/'legacy_export')
    parser.add_argument('--artifact-dir', type=Path, default=DATA/'sf0_diagnostic')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E009_h3_sf0_domain.json')
    args = parser.parse_args()
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError('Do not overwrite previous SF0 diagnostic evidence')
    args.artifact_dir.mkdir(parents=True)
    report = {'experiment': 'E009', 'scope': 'exact original native block1.fc2 SF0 failure diagnostic', 'status': 'running'}
    started = time.monotonic()
    try:
        execute(args, report)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total_not_benchmark'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        print(json.dumps({'status': report['status'], 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
