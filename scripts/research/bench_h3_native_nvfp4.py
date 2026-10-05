#!/usr/bin/env python3
"""E009: one fixed real H3 input, offload/resident/native correctness only.

Full block tensors are copied to CPU/disk for diagnostics. Timings from this
runner are deliberately not inference benchmarks. No sampling or video decode.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E009')
sys.path.insert(0, str(ROOT/'scripts'))
from minimax_h3_svdquant_common import (
    H3_DIT_PATH, TARGET_SUFFIXES, load_h3_pipeline, tree_device,
    install_runtime_hooks, nvfp4_qdq)
from probe_h3_modality_pulse import partitions, summarize, sha256, tensor_sha, save

TARGETS = tuple(f'blocks.{i}.{suffix}' for i in range(50) for suffix in TARGET_SUFFIXES)


def memory_guard(limit=60.):
    allocated = torch.cuda.memory_allocated()/1024**3
    if allocated > limit:
        raise RuntimeError(f'E009 memory gate exceeded: {allocated:.3f} GiB > {limit}')
    return allocated


def inventory(dit):
    entries = {}
    for kind, iterator in [('parameter', dit.named_parameters()), ('buffer', dit.named_buffers())]:
        for name, tensor in iterator:
            entries[name] = {'kind': kind, 'shape': list(tensor.shape),
                             'dtype': str(tensor.dtype), 'device': str(tensor.device),
                             'bytes': tensor.numel()*tensor.element_size()}
    return entries


@torch.inference_mode()
def make_h3_resident(dit):
    """Unwrap only the two verified H3 wrapper types, preserving other dtypes."""
    from diffsynth.core.vram.layers import AutoTorchModule, AutoWrappedLinear, AutoWrappedModule
    wrappers = {name: module for name, module in dit.named_modules()
                if isinstance(module, AutoTorchModule)}
    before = inventory(dit)
    unwrapped_names = tuple(wrappers)
    untouched = {name: row for name, row in before.items()
                 if not any(name == w or name.startswith(w+'.') for w in unwrapped_names)}
    report = {'wrapper_count': len(wrappers), 'replacements': [],
              'nonwrapper_dtypes_before': {k: v['dtype'] for k, v in untouched.items()}}
    # DiffSynth materializes the root table, but not this unwrapped child
    # Parameter. Its existing forward rebuilds frequencies dynamically. Load
    # the real saved value nonetheless; never initialize an empty replacement.
    report['materialized_nonwrapper_parameters'] = []
    for name, param in list(dit.named_parameters()):
        if name in untouched and param.is_meta:
            if name != 'rope.inv_freq':
                raise RuntimeError(f'Unreviewed nonwrapper meta parameter: {name}')
            from safetensors import safe_open
            with safe_open(str(H3_DIT_PATH), framework='pt', device='cpu') as checkpoint:
                saved = checkpoint.get_tensor(name)
            if saved.shape != param.shape or saved.dtype != param.dtype:
                raise RuntimeError(f'{name}: original checkpoint shape/dtype differs')
            parent, attr = name.rsplit('.', 1)
            setattr(dit.get_submodule(parent), attr,
                    nn.Parameter(saved.to('cuda'), requires_grad=param.requires_grad))
            report['materialized_nonwrapper_parameters'].append(
                {'name': name, 'source': str(H3_DIT_PATH), 'dtype': str(saved.dtype),
                 'shape': list(saved.shape), 'sha256': tensor_sha(saved)})
    # Keep no external reference to loaded old weights after replacing a module.
    for name, old in wrappers.items():
        if old._forward_hooks or old._forward_pre_hooks:
            raise RuntimeError(f'{name}: unexpected wrapper hook')
        if old.computation_dtype != torch.bfloat16 or not old.disk_offload:
            raise RuntimeError(f'{name}: unreviewed wrapper dtype/source')
        if isinstance(old, AutoWrappedLinear):
            if old.lora_A_weights or old.lora_B_weights or old.lora_merger is not None:
                raise RuntimeError(f'{name}: unexpected LoRA')
            weight, bias = old.load_from_disk(old.computation_dtype, 'cuda', assign=False)
            new = nn.Linear(old.in_features, old.out_features, bias=bias is not None,
                            device='meta', dtype=weight.dtype)
            new.weight = nn.Parameter(weight, requires_grad=False)
            if bias is not None:
                new.bias = nn.Parameter(bias, requires_grad=False)
            old.weight = old.bias = None
        elif isinstance(old, AutoWrappedModule) and isinstance(old.module, nn.RMSNorm):
            new = old.load_from_disk(old.computation_dtype, 'cuda', copy_module=False)
        else:
            raise RuntimeError(f'{name}: unsupported wrapper {type(old)}')
        new.train(old.training)
        parent, attr = name.rsplit('.', 1) if '.' in name else ('', name)
        setattr(dit.get_submodule(parent), attr, new)
        report['replacements'].append({'name': name, 'old_class': type(old).__name__,
                                       'new_class': type(new).__name__,
                                       'dtype': str(old.computation_dtype)})
        memory_guard()
    del wrappers
    # No dtype argument: e.g. timestep buffers must retain their actual dtype.
    dit.to(device='cuda')
    if any(isinstance(m, AutoTorchModule) for m in dit.modules()):
        raise RuntimeError('AutoTorchModule remains after resident migration')
    after = inventory(dit)
    if any(v['device'] != 'cuda:0' for v in after.values()):
        raise RuntimeError('Non-CUDA parameter/buffer after resident migration')
    for name, row in untouched.items():
        if after[name]['dtype'] != row['dtype'] or after[name]['shape'] != row['shape']:
            raise RuntimeError(f'{name}: nonwrapper tensor dtype/shape changed')
    if len(dit.blocks) != 50 or any(type(dit.get_submodule(n)) is not nn.Linear for n in TARGETS):
        raise RuntimeError('Unexpected main-block architecture after resident migration')
    refiner = [name for name, m in dit.named_modules()
               if 'token_refiner' in name and isinstance(m, nn.Linear)
               and any(name.endswith(suffix) for suffix in TARGET_SUFFIXES)]
    if len(refiner) != 8:
        raise RuntimeError(f'Expected eight refiner linears; got {refiner}')
    report.update(target_count=200, refiner_non_target_names=refiner,
                  tensor_bytes=sum(v['bytes'] for v in after.values()),
                  inventory_after=after, allocated_gib=memory_guard())
    return report


def non_target_hashes(dit):
    result = {}
    for name, tensor in list(dit.named_parameters())+list(dit.named_buffers()):
        if not any(name.startswith(target+'.') for target in TARGETS):
            result[name] = {'sha256': tensor_sha(tensor), 'dtype': str(tensor.dtype),
                            'shape': list(tensor.shape)}
    return result


class RuntimeAudit:
    def __init__(self):
        self.phase = 'setup'
        self.rows = {}

    def row(self):
        return self.rows.setdefault(self.phase, {'sdpa_calls': 0, 'scaled_mm_calls': 0,
                                                'disk_loads': 0, 'qkv_dtypes': []})

    @contextmanager
    def installed(self):
        from diffsynth.core.vram.layers import AutoWrappedLinear, AutoWrappedModule
        sdpa, mm = F.scaled_dot_product_attention, F.scaled_mm
        disk_functions = [(cls, cls.load_from_disk) for cls in (AutoWrappedLinear, AutoWrappedModule)]

        def audited_sdpa(q, k, v, *args, **kwargs):
            row = self.row()
            row['sdpa_calls'] += 1
            dtype = [str(t.dtype) for t in (q, k, v)]
            if dtype not in row['qkv_dtypes']:
                row['qkv_dtypes'].append(dtype)
            if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
                raise RuntimeError('E009 requires actual BF16 torch SDPA Q/K/V')
            return sdpa(q, k, v, *args, **kwargs)

        def audited_mm(a, b, *args, **kwargs):
            self.row()['scaled_mm_calls'] += 1
            if a.dtype != torch.float4_e2m1fn_x2 or b.dtype != torch.float4_e2m1fn_x2:
                raise RuntimeError('Non-FP4 scaled_mm observed')
            return mm(a, b, *args, **kwargs)

        def audited_disk(original):
            def wrapped(*args, **kwargs):
                self.row()['disk_loads'] += 1
                if self.phase in ('resident_bf16', 'legacy_formula_reference', 'native', 'four_linears'):
                    raise RuntimeError('Disk model load attempted during resident forward')
                return original(*args, **kwargs)
            return wrapped

        try:
            F.scaled_dot_product_attention, F.scaled_mm = audited_sdpa, audited_mm
            for cls, func in disk_functions:
                cls.load_from_disk = audited_disk(func)
            yield self
        finally:
            F.scaled_dot_product_attention, F.scaled_mm = sdpa, mm
            for cls, func in disk_functions:
                cls.load_from_disk = func


@torch.inference_mode()
def run_stage(dit, call, sample, stage, references, artifact_dir, report, output, audit,
              *, persist_blocks=False, require_exact=False):
    print(f'E009 stage={stage} starting', flush=True)
    audit.phase = stage
    stage_dir = artifact_dir/stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    row = {'blocks': [], 'directory': str(stage_dir)}
    report['stages'][stage] = row
    handles = []
    indices = None
    started = time.monotonic()

    def hook_for(index):
        def hook(_module, _inputs, out):
            nonlocal indices
            if not torch.is_tensor(out) or not bool(torch.isfinite(out).all()):
                raise RuntimeError(f'{stage} block{index}: invalid/nonfinite output')
            cpu = out.detach().cpu()
            if indices is None:
                indices, desc = partitions(sample, cpu.shape[0])
                report.setdefault('partitions', desc)
                indices['valid'] = torch.cat([indices[n] for n in ('video', 'audio', 'text')]).sort().values
            block = {'index': index, 'shape': list(cpu.shape), 'sha256': tensor_sha(cpu),
                     'finite': True, 'comparisons': {}}
            if persist_blocks:
                path = stage_dir/f'block_{index:02d}.pt'
                torch.save(cpu, path)
                block['tensor_file'] = str(path)
            for ref_stage in references:
                ref_info = report['stages'][ref_stage]['blocks'][index]
                ref = torch.load(ref_info['tensor_file'], map_location='cpu', weights_only=True, mmap=True)
                metrics = {'all': summarize(cpu, ref), 'bitwise_equal': block['sha256'] == ref_info['sha256']}
                # Resident migration requires equality, so equal partitions add no information.
                if not require_exact:
                    metrics.update({name: summarize(cpu, ref, idx) for name, idx in indices.items()})
                block['comparisons'][ref_stage] = metrics
                if require_exact and (metrics['all']['err2'] != 0 or not metrics['bitwise_equal']):
                    row['blocks'].append(block)
                    save(report, output)
                    raise RuntimeError(f'{stage} block{index}: resident/offload mismatch')
                del ref
            row['blocks'].append(block)
            memory_guard()
            if index % 10 == 0 or index == 49:
                save(report, output)
                print(f'E009 {stage} block={index} finite, allocated={torch.cuda.memory_allocated()/1024**3:.2f}GiB', flush=True)
        return hook

    try:
        for index, block in enumerate(dit.blocks):
            handles.append(block.register_forward_hook(hook_for(index)))
        result = dit(*call['input_args'], **call['input_kwargs'])
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError('Expected H3 (video, audio) output')
        if len(row['blocks']) != 50:
            raise RuntimeError('Did not observe exactly 50 main-block outputs')
        row['endpoints'] = {}
        cpu_outputs = {}
        for name, tensor in zip(('video', 'audio'), result, strict=True):
            if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f'{stage}: invalid {name} endpoint')
            cpu = tensor.detach().cpu()
            cpu_outputs[name] = cpu
            endpoint = {'shape': list(cpu.shape), 'sha256': tensor_sha(cpu), 'finite': True, 'comparisons': {}}
            for ref_stage in references:
                ref_path = report['stages'][ref_stage]['endpoints_file']
                ref = torch.load(ref_path, map_location='cpu', weights_only=True, mmap=True)[name]
                metric = summarize(cpu, ref)
                metric['bitwise_equal'] = endpoint['sha256'] == report['stages'][ref_stage]['endpoints'][name]['sha256']
                endpoint['comparisons'][ref_stage] = metric
                if require_exact and (metric['err2'] != 0 or not metric['bitwise_equal']):
                    raise RuntimeError(f'{stage} {name}: endpoint migration mismatch')
            row['endpoints'][name] = endpoint
        path = stage_dir/'endpoints.pt'
        torch.save(cpu_outputs, path)
        row['endpoints_file'] = str(path)
        row['endpoints_file_sha256'] = sha256(path)
    finally:
        for handle in handles:
            handle.remove()
        row['runtime_audit'] = dict(audit.row())
        row['seconds_including_diagnostics_not_benchmark'] = time.monotonic()-started
        row['peak_allocated_gib_so_far'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, output)
    calls = row['runtime_audit']['sdpa_calls']
    if calls < 50 or (stage != 'offload_bf16' and calls != report['stages']['offload_bf16']['runtime_audit']['sdpa_calls']):
        raise RuntimeError(f'{stage}: inconsistent actual torch BF16 SDPA count {calls}')
    expected_mm = 200 if stage == 'native' else 0
    if row['runtime_audit']['scaled_mm_calls'] != expected_mm:
        raise RuntimeError(f'{stage}: expected {expected_mm} FP4 GEMM calls')
    row['complete'] = True
    save(report, output)
    print(f'E009 stage={stage} complete endpoints={json.dumps(row["endpoints"])}', flush=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def check_four_linears(dit, raw_inputs, chunk_rows, report, output, audit):
    from h3_native_nvfp4 import native_execution_mode
    audit.phase = 'four_linears'
    report['four_representative_linears'] = []
    for suffix in TARGET_SUFFIXES:
        name = 'blocks.0.'+suffix
        module = dit.get_submodule(name)
        raw = raw_inputs[name].reshape(-1, module.in_features)[:512].cuda()
        weight = module.weight_packet().decode(chunk_rows=chunk_rows)
        ref = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None,
                        device='meta', dtype=torch.bfloat16)
        ref.weight = nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            ref.bias = nn.Parameter(module.bias, requires_grad=False)
        handles = install_runtime_hooks(ref, module.smooth, module.lr_a, module.lr_b, element_size=chunk_rows)
        try:
            expected = ref(raw)
        finally:
            handles.remove()
        with native_execution_mode(dit, 'legacy_qdq'):
            got = module(raw)
        hook_metric = summarize(got, expected)
        hook_metric['bitwise_equal'] = tensor_sha(got) == tensor_sha(expected)
        if hook_metric['err2'] != 0 or not hook_metric['bitwise_equal']:
            raise RuntimeError(f'{name}: legacy formula differs from independent common hooks')
        x_s = raw/module.smooth
        packet = module.pack_input(x_s)
        decoded = packet.decode(chunk_rows=chunk_rows)
        original_qdq = nvfp4_qdq(x_s, element_size=chunk_rows)
        # Native E005 codes retain the sign of values rounded to zero. The
        # common signed-15 codebook represents those as +0. This is the sole
        # permitted decode-byte difference; every nonzero value must be exact.
        activation_exact = torch.equal(decoded, original_qdq)
        byte_difference = decoded.contiguous().view(torch.int16) != original_qdq.contiguous().view(torch.int16)
        zero_sign_difference = byte_difference & (decoded == 0) & (original_qdq == 0)
        unexpected_byte_difference = int((byte_difference & ~zero_sign_difference).sum())
        if not activation_exact:
            raise RuntimeError(f'{name}: real activation packet does not decode exactly as legacy QDQ')
        if unexpected_byte_difference:
            raise RuntimeError(f'{name}: non-zero-sign activation byte difference')
        main_ref = module.main_from_packet(packet, mode='packed_qdq', include_bias=False)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            main_native = module.main_from_packet(packet, mode='native', include_bias=False)
            torch.cuda.synchronize()
        kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
        native_kernels = [name for name in kernels if 'sm120' in name.lower() and 'e2m1' in name.lower()]
        metric = summarize(main_native, main_ref)
        row = {'name': name, 'input_shape': list(raw.shape), 'independent_common_hook_output': hook_metric,
               'activation_decode_exact': activation_exact, 'samepacket_main': metric,
               'activation_decode_zero_sign_element_differences': int(zero_sign_difference.sum()),
               'activation_decode_unexpected_byte_differences': unexpected_byte_difference,
               'native_kernel_names': native_kernels, 'all_cuda_kernel_names': kernels}
        report['four_representative_linears'].append(row)
        save(report, output)
        if not native_kernels or not bool(torch.isfinite(main_native).all()) or metric['nmse'] > 1e-4:
            raise RuntimeError(f'{name}: representative native main gate failed')
        print(f'E009 representative {name} samepacket NMSE={metric["nmse"]:.8g}, common_hooks exact', flush=True)
        del raw, weight, ref, expected, got, x_s, packet, decoded, original_qdq, main_ref, main_native, handles
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def execute(args, report):
    torch.set_num_threads(6)
    torch.manual_seed(20261002)
    attn = importlib.import_module('diffsynth.core.attention.attention')
    if attn.ATTENTION_IMPLEMENTATION != 'torch':
        raise RuntimeError('Explicit DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch required')
    paths = [Path(__file__), args.plan, args.sample, args.state, H3_DIT_PATH,
             ROOT/'scripts/minimax_h3_svdquant_common.py', Path(__file__).with_name('probe_h3_modality_pulse.py')]
    for name in ['diffsynth.models.minimax_h3_dit', 'diffsynth.models.minimax_h3_dit_comfy',
                 'diffsynth.core.attention.attention', 'diffsynth.core.vram.layers']:
        paths.append(Path(importlib.import_module(name).__file__))
    if args.phase == 'full':
        paths.extend([Path(__file__).with_name('h3_native_nvfp4.py'), Path(__file__).with_name('wan_native_nvfp4.py')])
    print('E009 hashing model, fixed sample, state, plan and runtime sources', flush=True)
    report.update(files={str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in paths},
                  torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()),
                  attention_implementation=attn.ATTENTION_IMPLEMENTATION,
                  sdpa_enabled={'flash': torch.backends.cuda.flash_sdp_enabled(),
                                'math': torch.backends.cuda.math_sdp_enabled(),
                                'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
                                'cudnn': torch.backends.cuda.cudnn_sdp_enabled()},
                  environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'DIFFSYNTH_ROOT',
                               'MINIMAX_H3_DIT_PATH', 'DIFFSYNTH_ATTENTION_IMPLEMENTATION', 'SVDQUANT_DATA_ROOT']})
    if args.phase == 'full':
        manifest_path = args.export_dir/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] != 'complete':
            raise RuntimeError('Wait for a complete frozen export')
        checked = {}
        for path in [H3_DIT_PATH, args.state, ROOT/'scripts/minimax_h3_svdquant_common.py',
                     Path(__file__).with_name('h3_native_nvfp4.py'), Path(__file__).with_name('wan_native_nvfp4.py')]:
            expected = manifest['sources'][str(path)]['sha256']
            actual = report['files'][str(path)]['sha256']
            if expected != actual:
                raise RuntimeError(f'Export source SHA differs: {path}')
            checked[str(path)] = actual
        report['export_source_gates'] = checked
        report['export_manifest_sha256'] = sha256(manifest_path)
    save(report, args.output)
    sample = torch.load(args.sample, map_location='cpu', weights_only=False, mmap=True)
    if sample['input_kwargs'].get('control_hints') is not None:
        raise RuntimeError('Unexpected control hints')
    call = {'input_args': tree_device(sample['input_args'], 'cuda'),
            'input_kwargs': tree_device(sample['input_kwargs'], 'cuda')}
    with RuntimeAudit().installed() as audit:
        pipe = load_h3_pipeline(full=False, vram_limit_gib=30.)
        pipe.load_models_to_device(['dit'])
        dit = pipe.dit.eval()
        if len(dit.blocks) != 50:
            raise RuntimeError('Expected 50 main H3 blocks')
        run_stage(dit, call, sample, 'offload_bf16', [], args.artifact_dir, report, args.output, audit, persist_blocks=True)
        audit.phase = 'resident_migration'
        report['resident_conversion'] = make_h3_resident(dit)
        # Do not call any pipeline model onload/offload helper after conversion.
        run_stage(dit, call, sample, 'resident_bf16', ['offload_bf16'], args.artifact_dir, report,
                  args.output, audit, require_exact=True)
        report['resident_offload_exact'] = True
        save(report, args.output)
        if args.phase == 'resident':
            report['status'] = 'complete'
            report['scope_completed'] = 'resident/offload BF16 equivalence only; no native quantization'
            return
        from h3_native_nvfp4 import install_native_h3, NativeH3Linear, native_execution_mode
        before_non_targets = non_target_hashes(dit)
        report['non_target_tensors_before'] = before_non_targets
        audit.phase = 'install_native'
        report['native_installation'] = install_native_h3(dit, args.export_dir, device='cuda',
                                                         activation_packer='legacy', chunk_rows=args.chunk_rows)
        if sum(isinstance(m, NativeH3Linear) for m in dit.modules()) != 200:
            raise RuntimeError('Expected exactly 200 installed native targets')
        if non_target_hashes(dit) != before_non_targets:
            raise RuntimeError('Native conversion changed a non-target tensor')
        gc.collect()
        torch.cuda.empty_cache()
        raw_inputs, smooth_inputs, handles = {}, {}, []
        for suffix in TARGET_SUFFIXES:
            name = 'blocks.0.'+suffix
            module = dit.get_submodule(name)
            def capture(mod, inputs, *, key=name):
                if key in raw_inputs:
                    raise RuntimeError(f'Duplicate representative capture {key}')
                raw_inputs[key] = inputs[0].detach().cpu()
                smooth_inputs[key] = (inputs[0]/mod.smooth).detach().cpu()
            handles.append(module.register_forward_pre_hook(capture))
        try:
            with native_execution_mode(dit, 'legacy_qdq'):
                run_stage(dit, call, sample, 'legacy_formula_reference', ['offload_bf16'], args.artifact_dir,
                          report, args.output, audit, persist_blocks=True)
        finally:
            for handle in handles:
                handle.remove()
        pack_path = DATA/'pack_inputs.pt'
        if pack_path.exists():
            raise FileExistsError(f'Refusing to overwrite capture {pack_path}')
        torch.save({'scope': 'p1 step0 PTQ-calib; actual legacy full trajectory block0',
                    'capture_location': 'after BF16 smoothing, before QDQ; complete original token length',
                    'sample_sha256': report['files'][str(args.sample)]['sha256'], 'inputs': smooth_inputs}, pack_path)
        report['pack_inputs'] = {'file': str(pack_path), 'sha256': sha256(pack_path),
                                 'shapes': {k: list(v.shape) for k, v in smooth_inputs.items()}}
        del smooth_inputs
        save(report, args.output)
        print(f'E009 full real smoothed inputs saved: {pack_path}', flush=True)
        check_four_linears(dit, raw_inputs, args.chunk_rows, report, args.output, audit)
        del raw_inputs
        with native_execution_mode(dit, 'native'):
            run_stage(dit, call, sample, 'native', ['offload_bf16', 'legacy_formula_reference'], args.artifact_dir,
                      report, args.output, audit)
        if non_target_hashes(dit) != before_non_targets:
            raise RuntimeError('Forward changed a non-target tensor')
        report['non_target_tensors_unchanged_after_all_forwards'] = True
        report['runtime_audit'] = audit.rows
        report['status'] = 'complete'
        report['scope_completed'] = 'one original-calib p1 step0 complete resident BF16/legacy-formula/native correctness'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['resident', 'full'], required=True)
    parser.add_argument('--sample', type=Path, default=ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt')
    parser.add_argument('--state', type=Path, default=ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt')
    parser.add_argument('--plan', type=Path, default=ROOT/'research_state/06_experiments/E009_h3_native_baseline_plan.md')
    parser.add_argument('--export-dir', type=Path, default=DATA/'legacy_export')
    parser.add_argument('--artifact-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--chunk-rows', type=int, default=1024)
    args = parser.parse_args()
    if args.sample != ROOT/'results/calib/minimax_h3_svdquant_standard_8p64s/p1/sample_p1_s00.pt':
        parser.error('E009 freezes p1 step0; do not substitute another sample')
    if args.chunk_rows <= 0:
        parser.error('chunk rows must be positive')
    args.artifact_dir = args.artifact_dir or DATA/f'p1s00_{args.phase}'
    args.output = args.output or ROOT/f'results/research/E009_h3_{args.phase}.json'
    if args.output.exists() or args.artifact_dir.exists():
        raise FileExistsError('Preserve previous artifacts; choose a new output AND artifact directory')
    args.artifact_dir.mkdir(parents=True)
    report = {'experiment': 'E009', 'status': 'running', 'phase': args.phase,
              'implementation_revision': 2,
              'scope': 'fixed original-PTQ-calib prompt1 step0, correctness only',
              'legacy_reference': 'common activation QDQ + exported exact legacy W + corrected BF16 LR; independently checked four real linears',
              'chunk_rows': args.chunk_rows, 'stages': {}}
    started = time.monotonic()
    try:
        execute(args, report)
    except BaseException as exc:
        report['status'] = 'failed_stop'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['seconds_total_including_hashing_io_diagnostics_not_benchmark'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        save(report, args.output)
        print(json.dumps({'status': report['status'], 'report': str(args.output),
                          'seconds': report['seconds_total_including_hashing_io_diagnostics_not_benchmark']}), flush=True)


if __name__ == '__main__':
    main()
