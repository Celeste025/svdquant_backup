#!/usr/bin/env python3
"""Prepared full Wan14B PTQ worker; no experiment or execution budget is assigned here.

Reuse E052's full numerical configuration and40/400registration, E047's ungated
RoPE compatibility, and the original full-model PTQ phase order. No partial resume.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import traceback

import ptq_wan_matched_calibration as base
import wan14b_ptq_resource as resource_pilot

ROOT = base.ROOT


def check_inputs(args, m, report):
    import torch
    from diffusers import WanTransformer3DModel
    cm = json.loads(resource_pilot.checked_record(m['collection_manifest']).read_text())
    base.require(cm['experiment'] == 'E050' and cm['model_dir'] == m['model_dir'] and
                 cm['settings'] == m['settings'] and cm['selected_caches'] == m['selected_caches'],
                 'Expected the64 actual matched14B E050 records')
    # Only relocate output/cache directories. No numerical recipe field is changed.
    config_input = dict(m)
    config_input['data_dir'] = str(args.run_dir or Path(m['data_dir']) / 'unlaunched_full_ptq')
    config, report['configuration'] = resource_pilot.parse_config(config_input)
    report['configuration']['inherited_yaml_note'] = (
        'E052 numerical configuration reused with an explicit14B name/path/81f/CFG5/shift3 protocol. '
        'This worker calls the original full-model PTQ; no scheduler or generation is run.')
    mc = WanTransformer3DModel.load_config(m['model_dir'], subfolder='transformer', local_files_only=True)
    with torch.device('meta'):
        meta = WanTransformer3DModel.from_config(mc)
    structured, report['meta_registration'] = resource_pilot.register_wan(meta)
    layers, _, recomputes, use_prev = structured.get_iter_layer_activations_args(
        skip_pre_modules=True, skip_post_modules=True)
    base.require(len(layers) == 40 and layers == list(meta.blocks) and
                 recomputes == [False] * 40 and use_prev == [False] + [True] * 39,
                 'Full40-layer loader must remain untruncated')
    del meta, structured, layers
    report['resource_risks'] = dict(
        full_model_info_pass='Each fresh activation-loader traversal executes all64 inputs through all40 blocks '
                             'before yielding block0; E052 stopped after block0 instead.',
        caches='Layer activations are cached on CPU; full-model forward kwargs/previous outputs, loader workers '
               'and calibration reference outputs consume memory beyond resident weights.',
        later_layers='Different low-rank early-stop lengths, data-dependent temporaries and accumulating branch '
                     'state can exceed block0 time or memory. No linear extrapolation is made.',
        final_export='Full weight quantization, retained scales/branch state and dense model.pt serialization '
                     'are outside E052. Checkpoint remains linked to its fresh cache when copy_on_save=False.',
        clone_original_weights_for_activation_calibration=config.quant.needs_acts_quantizer_cache,
        weight_range_requires_data=config.quant.wgts.needs_calib_data,
        gptq_enabled=config.quant.wgts.enabled_gptq)
    return config


def run(args, m, config, report):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import WanTransformer3DModel
    import deepcompressor.app.diffusion.dataset.calib as calib
    from deepcompressor.app.diffusion.ptq import ptq
    from deepcompressor.utils import tools
    base.require(args.deadline_unix is not None and time.time() < args.deadline_unix,
                 'An explicit execution deadline selected after the resource result is required')
    base.require(args.run_dir is not None and args.checkpoint_dir is not None,
                 'Run requires explicit fresh --run-dir and --checkpoint-dir')
    base.require(not args.run_dir.exists() and not args.checkpoint_dir.exists(), 'Preserve prior output/cache/checkpoint')
    base.require(args.run_dir.resolve() != args.checkpoint_dir.resolve(), 'Separate run and checkpoint directories')
    pilot = json.loads(args.resource_report.read_text())
    base.require(pilot['experiment'] == 'E052' and pilot['phase'] == 'run' and pilot['status'] == 'complete',
                 'Wait for the complete E052 resource result and a subsequent execution-budget decision')
    pilot_manifest = json.loads(resource_pilot.checked_record(pilot['manifest']).read_text())
    base.require(pilot_manifest['model_dir'] == m['model_dir'] and
                 pilot_manifest['collection_manifest'] == m['collection_manifest'] and
                 pilot_manifest['ptq'] == m['ptq'], 'Resource pilot must cover this model/data/recipe')
    report['resource_pilot'] = base.record(args.resource_report)
    report['collection'] = resource_pilot.collection_ready(m)
    report['partial_artifacts_loaded'] = False
    base.require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.run_dir.mkdir(parents=True)
    tools.logging.setup(path=str(args.run_dir / 'run.log'), level=tools.logging.DEBUG)
    config.dump(path=str(args.run_dir / 'config.yaml'))
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                          visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    started = time.monotonic()
    model = WanTransformer3DModel.from_pretrained(m['model_dir'], subfolder='transformer',
        torch_dtype=torch.bfloat16, local_files_only=True).eval().to('cuda')
    structured, report['registration'] = resource_pilot.register_wan(model)
    base.require(all(p.device.type == 'cuda' for p in model.parameters()), 'Full transformer must stay on GPU')
    torch.cuda.synchronize()
    report.update(model_weights_loaded=True, model_load_seconds=time.monotonic()-started,
        model_loads=dict(transformer=1, text_encoder=0, vae=0), loaded_model_config=dict(model.config),
        resident_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        resident_allocated_bytes=torch.cuda.memory_allocated(), resident_reserved_bytes=torch.cuda.memory_reserved())
    torch.manual_seed(m['random_seed'])
    # This patch only suppresses the historical gate cache and preserves RoPE tuples.
    # It does not restrict the layer lists or install early_stop_module.
    old_attach, old_iter = base.install_compatibility(args, report)
    base.save(args.output, report)
    started = time.monotonic()
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ptq(structured, config.quant, cache=config.cache, load_dirpath='',
                save_dirpath=str(args.checkpoint_dir), copy_on_save=False, save_model=True)
        torch.cuda.synchronize(); base.deadline(args)
    finally:
        calib._attach_wan_gate_eval_kwargs = old_attach
        calib.DiffusionCalibCacheLoader.iter_layer_activations = old_iter
        report.update(ptq_seconds_including_original_cache_and_export=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            loader_yields_by_layer=dict(Counter(row['layer'] for row in report['layer_events'])))
        base.save(args.output, report)
    files = [base.record(args.checkpoint_dir / f'{name}.pt')
             for name in ('model', 'scale', 'wgts', 'branch', 'smooth')]
    identity = dict(status='complete', operation='wan14b_matched_full_ptq', experiment=m.get('experiment'),
        model_dir=m['model_dir'], manifest=report['manifest'], source=report['source'],
        files=files, collection=report['collection'], resource_pilot=report['resource_pilot'],
        recipe='real_nvfp4 + wan_s16;64/b4/g10/rank32/LR100 early-stop; original full-model phase order',
        partial_artifacts_loaded=False)
    base.save(args.run_dir / 'checkpoint_identity.json', identity)
    report['checkpoint_identity'] = base.record(args.run_dir / 'checkpoint_identity.json')
    report['final_memory'] = resource_pilot.memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'run'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--resource-report', type=Path, default=ROOT / 'results/research/E052/run.json')
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    base.require(not args.output.exists(), 'Preserve earlier result; specify a new output explicitly')
    m = json.loads(args.manifest.read_text())
    report = dict(operation='wan14b_matched_full_ptq',
        experiment=m.get('experiment') if args.phase == 'run' else None,
        manifest_experiment=m.get('experiment'), prepared_only=args.phase == 'check', phase=args.phase,
        status='running', pid=os.getpid(), start_epoch=time.time(), source=base.record(Path(__file__)),
        manifest=base.record(args.manifest), layer_events=[], model_weights_loaded=False,
        output_directory_contract=dict(run_dir=str(args.run_dir) if args.run_dir else None,
            checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir else None),
        execution_note='Prepared worker only. Execution budget/launcher/protocol are separate; '
                       'a complete resource pilot alone is not a full-model feasibility guarantee.')
    report['sources'] = [report['source'], report['manifest'], base.record(Path(base.__file__)),
        base.record(Path(resource_pilot.__file__)), base.record(base.DC / 'deepcompressor/app/diffusion/ptq.py')]
    try:
        os.environ['DEEPCOMPRESSOR_WAN_GATED'] = '0'
        import torch
        import pyarrow  # noqa: F401; preserve the successful E047 import order
        import diffusers
        torch.set_num_threads(4)
        if args.phase == 'check':
            base.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available(), 'Hide CUDA')
        report['environment'] = dict(python=sys.executable, torch=torch.__version__, diffusers=diffusers.__version__,
                                    cuda=torch.version.cuda)
        config = check_inputs(args, m, report)
        if args.phase == 'run':
            run(args, m, config, report)
        else:
            base.require(not torch.cuda.is_initialized(), 'CUDA initialized in CPU check')
            report.update(cuda_initialized=False, collection_completion_checked=False, resource_completion_checked=False,
                          check_note='Config/meta only; no completed E050/E052 claim, weights, calibration data or GPU execution.')
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc()); raise
    finally:
        report['seconds'] = time.time()-report['start_epoch']; base.save(args.output, report)
        print(report['status'], args.output, flush=True)


if __name__ == '__main__':
    main()
