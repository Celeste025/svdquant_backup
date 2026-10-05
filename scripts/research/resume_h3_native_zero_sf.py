#!/usr/bin/env python3
"""E009 explicit zero-SF compatibility: reuse frozen refs, run slow+fast native once."""
from __future__ import annotations
import argparse
import copy
import gc
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E009')
sys.path.insert(0, str(ROOT/'scripts'))
from minimax_h3_svdquant_common import load_h3_pipeline, tree_device, TARGET_SUFFIXES
from probe_h3_modality_pulse import sha256, tensor_sha, save
from bench_h3_native_nvfp4 import make_h3_resident, RuntimeAudit, run_stage, non_target_hashes, memory_guard
from h3_native_nvfp4 import install_native_h3, NativeH3Linear
from h3_nvfp4_zero_sf_compat import (
    COMPATIBILITY, pack_activation_legacy_zero_sf, pack_activation_fast_zero_sf, collect_fastpack_checks)


@torch.inference_mode()
def fast_replay(dit, call, report, args, audit):
    audit.phase = 'nativefast'
    row = {'status': 'running', 'blocks': [], 'runtime_audit': {}, 'endpoints': {},
           'reference': 'same-process new slow native compatibility pass; no teacher injection'}
    report['fast_proof'] = row
    handles = []
    started = time.monotonic()
    for module in dit.modules():
        if isinstance(module, NativeH3Linear):
            module.activation_packer = pack_activation_fast_zero_sf

    def hook_for(index):
        def hook(_module, _inputs, out):
            digest = tensor_sha(out)
            expected = report['stages']['native']['blocks'][index]['sha256']
            info = {'index': index, 'sha256': digest, 'slow_sha256': expected,
                    'bitwise_equal': digest == expected, 'finite': bool(torch.isfinite(out).all())}
            row['blocks'].append(info)
            if not info['bitwise_equal'] or not info['finite']:
                raise RuntimeError(f'fast block{index}: mismatch against slow native')
            memory_guard()
        return hook

    try:
        for index, block in enumerate(dit.blocks):
            handles.append(block.register_forward_hook(hook_for(index)))
        with collect_fastpack_checks() as checks:
            result = dit(*call['input_args'], **call['input_kwargs'])
            torch.cuda.synchronize()
        # Accept no output until all original invalid/nonfinite flags checked.
        row['fastpack_checks'] = checks.summary
        row['checks'] = {'summary': checks.summary}
        if checks.summary['checked_calls'] != 200:
            raise RuntimeError('Expected 200 checked fast packets')
        if len(row['blocks']) != 50 or not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError('Incomplete fast forward')
        cpu_outputs = {}
        for name, tensor in zip(('video', 'audio'), result, strict=True):
            cpu_outputs[name] = tensor.detach().cpu()
            digest = tensor_sha(tensor)
            expected = report['stages']['native']['endpoints'][name]['sha256']
            row['endpoints'][name] = {'sha256': digest, 'slow_sha256': expected,
                                     'bitwise_equal': digest == expected,
                                     'shape': list(tensor.shape), 'finite': bool(torch.isfinite(tensor).all())}
            if digest != expected or not row['endpoints'][name]['finite']:
                raise RuntimeError(f'fast {name}: mismatch against slow native')
        path = args.artifact_dir/'nativefast_endpoints.pt'
        torch.save(cpu_outputs, path)
        row['endpoints_file'] = str(path)
        row['endpoints_file_sha256'] = sha256(path)
        calls = audit.row()
        if calls['scaled_mm_calls'] != 200 or calls['disk_loads'] != 0 or calls['sdpa_calls'] != 102:
            raise RuntimeError(f'Unexpected actual fast execution: {calls}')
        row['status'] = 'complete'
    finally:
        for handle in handles:
            handle.remove()
        row['runtime_audit'] = dict(audit.row())
        row['seconds_including_diagnostics_not_benchmark'] = time.monotonic()-started
        save(report, args.output)


@torch.inference_mode()
def execute(args, report):
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    old = json.loads(args.reference.read_text())
    proof = json.loads(args.domain_proof.read_text())
    if old['status'] != 'failed_stop' or not old['resident_offload_exact']:
        raise RuntimeError('Expected original strict-domain failure with exact resident proof')
    parity = proof['legacy_qdq_vs_e005_decode']
    if (proof['status'] != 'complete' or proof['source_failure_sha256'] != sha256(args.reference)
            or not parity['torch_equal'] or parity['other_byte_differences'] != 0
            or not parity['finite_common'] or not parity['finite_decoded']):
        raise RuntimeError('Real failing-input zero-SF proof missing or incompatible')
    if proof['block0_sha256'] != old['stages']['native']['blocks'][0]['sha256']:
        raise RuntimeError('Domain proof did not reproduce the failed prefix')
    print('E009 resume validating inherited source hashes; no BF16 or QDQ full pass repeated', flush=True)
    # Include model, state, runner, plan, helper, raw sample and DiffSynth sources.
    # The extra zero-SF adapter is explicitly new and hashed below.
    for path, info in old['files'].items():
        if sha256(path) != info['sha256']:
            raise RuntimeError(f'Inherited source changed: {path}')
    report['inherited_reference'] = {'file': str(args.reference), 'sha256': sha256(args.reference),
                                    'stages': ['offload_bf16', 'resident_bf16', 'legacy_formula_reference'],
                                    'not_reexecuted': True, 'source_files': old['files']}
    report['source_failure_sha256'] = sha256(args.reference)
    report['domain_proof'] = {'file': str(args.domain_proof), 'sha256': sha256(args.domain_proof)}
    sources = [Path(__file__), Path(__file__).with_name('h3_nvfp4_zero_sf_compat.py'),
               ROOT/'research_state/06_experiments/E009_h3_zero_sf_resume_plan.md',
               Path(__file__).with_name('h3_nvfp4_fastpack.py'),
               Path(__file__).with_name('probe_h3_native_contract.py'),
               Path(__file__).with_name('smoke_nvfp4_systems_20261002.py'),
               ROOT/'results/research/E009_fastpack_legacy_four.json']
    report['files'] = {str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sources}
    for key in report['inherited_reference']['stages']:
        report['stages'][key] = copy.deepcopy(old['stages'][key])
        report['stages'][key]['inherited_from'] = str(args.reference)
    report['partitions'] = old['partitions']
    report['four_representative_linears_inherited'] = old['four_representative_linears']
    report['torch'], report['cuda'] = torch.__version__, torch.version.cuda
    if report['torch'] != old['torch'] or report['cuda'] != old['cuda']:
        raise RuntimeError('Runtime differs from inherited reference')
    report['device'] = torch.cuda.get_device_name()
    report['compatibility'] = COMPATIBILITY
    attention = importlib.import_module('diffsynth.core.attention.attention')
    if attention.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('Explicit original torch attention required')
    report['sdpa_enabled'] = {'flash': torch.backends.cuda.flash_sdp_enabled(),
                              'math': torch.backends.cuda.math_sdp_enabled(),
                              'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
                              'cudnn': torch.backends.cuda.cudnn_sdp_enabled()}
    if report['sdpa_enabled'] != old['sdpa_enabled']:
        raise RuntimeError('SDPA backend settings changed')
    if sha256(args.export_dir/'manifest.json') != old['export_manifest_sha256']:
        raise RuntimeError('Export manifest changed')
    # Verify the new fast adapter on the exact previously rejected full-shape
    # input before spending another full-model pass in the expanded domain.
    capture_path = Path(proof['capture']['file'])
    if sha256(capture_path) != proof['capture']['sha256']:
        raise RuntimeError('Failure capture changed')
    failure_x = torch.load(capture_path, map_location='cpu', weights_only=True, mmap=True)['x_s'].cuda()
    slow_packet = pack_activation_legacy_zero_sf(failure_x, chunk_rows=1024)
    with collect_fastpack_checks() as failure_checks:
        fast_packet = pack_activation_fast_zero_sf(failure_x)
    differences = {key: int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())
                   for key, a, b in [('codes', slow_packet.packed, fast_packet.packed),
                                     ('global', slow_packet.global_scale, fast_packet.global_scale),
                                     ('swizzled_sf_including_padding', slow_packet.swizzled_scales, fast_packet.swizzled_scales)]}
    report['failure_input_fast_byte_gate'] = {'byte_mismatches': differences,
                                             'checks': failure_checks.summary,
                                             'slow_statistics': slow_packet.zero_sf_statistics,
                                             'x_s_sha256': proof['capture']['x_s_sha256']}
    save(report, args.output)
    if (any(differences.values()) or failure_checks.summary['checked_calls'] != 1
            or failure_checks.summary['invalid_calls'] != 0
            or failure_checks.summary['calls_with_nonzero_input_zero_sf'] != 1):
        raise RuntimeError('Fast adapter fails exact real zero-SF packet/flag gate')
    del failure_x, slow_packet, fast_packet, failure_checks
    gc.collect()
    torch.cuda.empty_cache()
    sample = torch.load(args.sample, map_location='cpu', weights_only=False, mmap=True)
    if sha256(args.sample) != old['files'][str(args.sample)]['sha256']:
        raise RuntimeError('Fixed sample changed')
    call = {'input_args': tree_device(sample['input_args'], 'cuda'),
            'input_kwargs': tree_device(sample['input_kwargs'], 'cuda')}
    report['slow_zero_sf_calls'] = []
    with RuntimeAudit().installed() as audit:
        pipe = load_h3_pipeline(full=False, vram_limit_gib=30.)
        pipe.load_models_to_device(['dit'])
        dit = pipe.dit.eval()
        audit.phase = 'resident_migration'
        report['resident_conversion_reexecuted'] = make_h3_resident(dit)
        before = non_target_hashes(dit)
        if before != old['non_target_tensors_before']:
            raise RuntimeError('New resident non-target tensors differ from original reference')
        report['native_installation'] = install_native_h3(
            dit, args.export_dir, activation_packer=pack_activation_legacy_zero_sf, chunk_rows=1024)
        for name, module in dit.named_modules():
            if not isinstance(module, NativeH3Linear):
                continue
            def tracked_slow(x, *, chunk_rows=1024, layer=name):
                packet = pack_activation_legacy_zero_sf(x, chunk_rows=chunk_rows)
                report['slow_zero_sf_calls'].append({'name': layer, 'statistics': packet.zero_sf_statistics})
                return packet
            module.activation_packer = tracked_slow
        gc.collect()
        torch.cuda.empty_cache()
        expected_first = old['stages']['native']['blocks'][0]['sha256']
        def first_gate(_module, _inputs, output):
            digest = tensor_sha(output)
            report['strict_domain_prefix_block0_sha256'] = digest
            if digest != expected_first:
                raise RuntimeError('Zero-SF variant differs before first original domain failure')
        handle = dit.blocks[0].register_forward_hook(first_gate)
        try:
            run_stage(dit, call, sample, 'native', ['offload_bf16', 'legacy_formula_reference'],
                      args.artifact_dir, report, args.output, audit)
        finally:
            handle.remove()
        if len(report['slow_zero_sf_calls']) != 200:
            raise RuntimeError('Expected 200 real slow packets')
        report['slow_native_complete'] = True
        save(report, args.output)
        print('E009 zero-SF slow native complete; starting one full fast SHA replay', flush=True)
        fast_replay(dit, call, report, args, audit)
        if non_target_hashes(dit) != before:
            raise RuntimeError('Native slow/fast altered non-target tensors')
        report['non_target_tensors_unchanged'] = True
        report['runtime_audit'] = audit.rows
        report['status'] = 'complete'
        report['interpretation'] = ('Complete one-calib-input native baseline and byte-exact slow/fast replay. '
                                    'SF0 behavior is an explicit validated compatibility domain extension; no speed claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT/'results/research/E009_h3_full.json')
    parser.add_argument('--domain-proof', type=Path, default=ROOT/'results/research/E009_h3_sf0_domain_v2.json')
    parser.add_argument('--sample', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt')
    parser.add_argument('--export-dir', type=Path, default=DATA/'legacy_export')
    parser.add_argument('--artifact-dir', type=Path, default=DATA/'native_zero_sf_resume')
    parser.add_argument('--output', type=Path, default=ROOT/'results/research/E009_h3_native_resume.json')
    args = parser.parse_args()
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError('Refusing to overwrite prior native resume evidence')
    args.artifact_dir.mkdir(parents=True)
    report = {'experiment': 'E009', 'status': 'running', 'stages': {},
              'scope': 'one original-calib p1step0; only new native slow/fast passes, inherited BF16/QDQ refs'}
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
