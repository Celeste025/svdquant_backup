#!/usr/bin/env python3
"""E047: original-Wan PTQ on the frozen matched 64-record calibration set.

Only compatibility change: ungated Wan calibration does not build the obsolete
gate cache, and attention replay retains Diffusers 0.40's (cos, sin) tuple.
No sampling, smoothing-grid, rank, quantization, or layer-coverage changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DC = ROOT / 'third_party/deepcompressor'
EXAMPLES = DC / 'examples/diffusion'
MANIFEST = ROOT / 'research_state/06_experiments/E047_wan_matched_calibration_manifest.json'
PLAN = ROOT / 'research_state/06_experiments/E047_wan_matched_calibration_plan.md'
sys.path[:0] = [str(ROOT / 'scripts'), str(DC)]


def require(value, message):
    if not value:
        raise AssertionError(message)


def record(path):
    path = Path(path).absolute()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    st = path.stat()
    return dict(file=str(path), resolved=str(path.resolve()), bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, sha256=h.hexdigest())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def deadline(args):
    if args.deadline_unix is not None and time.time() >= args.deadline_unix:
        raise TimeoutError('E047 PTQ deadline')


def parse_config(m):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    config_files = ['configs/model/wan2.1-1.3b.yaml', 'configs/svdquant/real_nvfp4.yaml',
                    'configs/svdquant/wan_s16.yaml']
    argv = [*config_files, '--pipeline-path=' + m['model_dir'],
            '--calib-path=' + str(Path(m['data_dir']) / 'calibration'),
            '--calib-num-samples=64', '--output-root=' + m['ptq']['run_dir'],
            '--cache-root=' + str(Path(m['ptq']['run_dir']) / 'fresh_cache'),
            '--save-model=' + m['ptq']['checkpoint_dir'], '--skip-eval', '--skip-gen',
            '--eval-num-gpus=1', '--eval-batch-size-per-gpu=1', '--eval-num-frames=81',
            '--eval-protocol=unipc50-g6.0-f81-shift8']
    cwd, old_argv = Path.cwd(), sys.argv
    try:
        os.chdir(EXAMPLES); sys.argv = ['E047-ptq', *argv]
        config, _, unused_configs, unused_args, unknown = DiffusionPtqRunConfig.get_parser().parse_known_args()
        require(not unknown, f'Unknown config arguments: {unknown}')
    finally:
        os.chdir(cwd); sys.argv = old_argv
    q = config.quant
    require(q.calib.num_samples == 64 and q.calib.batch_size == 4, 'Calibration count/batch changed')
    require(q.wgts.low_rank.rank == 32 and q.wgts.low_rank.num_iters == 100 and
            q.wgts.low_rank.early_stop, 'Low-rank recipe changed')
    require(q.smooth.proj.num_grids == 10 and q.smooth.proj.sample_size == -1 and
            q.smooth.proj.sample_batch_size == 4, 'Smoothing recipe changed')
    require(q.wgts.low_rank.sample_size == -1 and q.wgts.low_rank.sample_batch_size == 4,
            'Low-rank data restriction changed')
    require(not config.pipeline.shift_activations and not q.ipts.static, 'Shift/static recipe changed')
    import torch
    for quant in (q.wgts, q.ipts):
        require(str(quant.dtype) == 'sfp4_e2m1_all', 'Not the original E2M1 recipe')
        require(quant.scale_dtypes[0] == torch.float32 and str(quant.scale_dtypes[1]) == 'sfp8_e4m3_nan',
                'Two-level FP32/E4M3 scales changed')
        require(quant.group_shapes[-1][1] == 16, 'Group size changed')
    return config, dict(argv=argv, unused_configs=unused_configs, unused_args=unused_args,
                        config_files=[record(EXAMPLES / p) for p in config_files],
                        resolved_config=config.dump())


def register_wan(model):
    from deepcompressor.app.diffusion.nn.struct import DiffusionAttentionStruct, DiffusionModelStruct
    import deepcompressor.app.diffusion.dataset.calib as calib
    kind = type(model.blocks[0].attn1)
    if kind not in DiffusionAttentionStruct._factories:
        DiffusionAttentionStruct.register_factory(kind, DiffusionAttentionStruct._default_construct)
    if kind not in calib._ATTN_TYPES:
        calib._ATTN_TYPES = (*calib._ATTN_TYPES, kind)
    structured = DiffusionModelStruct.construct(model)
    require(structured.module is model and len(model.blocks) == 30, 'Direct transformer construction failed')
    targets = [n for n, layer in model.named_modules() if n.startswith('blocks.') and
               isinstance(layer, __import__('torch').nn.Linear)]
    require(len(targets) == 300, 'Unexpected main-block linear coverage')
    return structured, dict(attention_type=f'{kind.__module__}.{kind.__name__}',
                            direct_transformer=True, blocks=30, target_count=len(targets), targets=targets)


def install_compatibility(args, report):
    import torch
    import deepcompressor.app.diffusion.dataset.calib as calib
    from deepcompressor.app.diffusion.quant.utils import _wan_gated_enabled
    require(os.environ.get('DEEPCOMPRESSOR_WAN_GATED') == '0' and not _wan_gated_enabled(),
            'This adapter is valid only for disabled historical gated calibration')
    old_attach = calib._attach_wan_gate_eval_kwargs
    old_iter = calib.DiffusionCalibCacheLoader.iter_layer_activations

    def no_gate(*unused, **unused_kwargs):
        require(not _wan_gated_enabled(), 'Gate was enabled during ungated PTQ')

    def iterator(self, model, *a, **kw):
        for name, (structure, cache, kwargs) in old_iter(self, model, *a, **kw):
            deadline(args)
            if 'rotary_emb' in kwargs:
                rot = kwargs['rotary_emb']
                require(isinstance(rot, tuple) and len(rot) == 2 and
                        all(isinstance(t, torch.Tensor) for t in rot), 'Expected 0.40 cosine/sine tuple')
                device = next(structure.module.parameters()).device
                before = [str(t.device) for t in rot]
                require(rot[0].shape == rot[1].shape and rot[0].shape[0] == 1 and rot[0].shape[2] == 1,
                        'Unexpected rotary broadcasting shape')
                kwargs['rotary_emb'] = tuple(t if t.device == device else t.to(device=device) for t in rot)
                event = dict(layer=name, rotary_shapes=[list(t.shape) for t in rot],
                             rotary_dtype=[str(t.dtype) for t in rot], original_devices=before,
                             evaluation_device=str(device), moved=[t.device != device for t in rot])
            else:
                event = dict(layer=name, rotary_present=False)
            event.update(elapsed_seconds=time.time()-report['start_epoch'],
                         allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            report['layer_events'].append(event); save(args.output, report)
            print('E047 layer ready', event, flush=True)
            yield name, (structure, cache, kwargs)
            deadline(args)

    calib._attach_wan_gate_eval_kwargs = no_gate
    calib.DiffusionCalibCacheLoader.iter_layer_activations = iterator
    return old_attach, old_iter


def collection_ready(m):
    rd = Path(m['report_dir'])
    evidence = {}
    for name in ('collect_launcher.json', 'collection_summary.json'):
        path = rd / name; value = json.loads(path.read_text())
        require(value['status'] == 'complete', f'Collection not complete: {name}')
        evidence[name] = record(path)
    cache_dir = Path(m['data_dir']) / 'calibration/caches'
    expected = {row['filename'] for row in m['selected_caches']}
    actual = {p.name for p in cache_dir.glob('*.pt')}
    require(len(expected) == 64 and actual == expected, 'Expected exactly the frozen 64 caches')
    evidence['cache_directory'] = str(cache_dir)
    evidence['cache_files'] = [dict(file=str(cache_dir / name), bytes=(cache_dir / name).stat().st_size)
                               for name in sorted(expected)]
    return evidence


def run(args, m, config, report):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import WanTransformer3DModel
    from deepcompressor.app.diffusion.ptq import ptq
    from deepcompressor.utils import tools
    deadline(args); report['collection'] = collection_ready(m)
    require(torch.cuda.is_available(), 'CUDA unavailable')
    checkpoint = Path(m['ptq']['checkpoint_dir']); run_dir = Path(m['ptq']['run_dir'])
    require(not checkpoint.exists() and not run_dir.exists(), 'Preserve existing PTQ/cache artifacts')
    torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    config.output.lock()
    try:
        config_path = Path(config.output.get_running_job_path('config.yaml')); config.dump(path=str(config_path))
        tools.logging.setup(path=config.output.get_running_job_path('run.log'), level=tools.logging.DEBUG)
        print(config.formatted_str(), flush=True)
        report['config_running_path'] = str(config_path)
        model = WanTransformer3DModel.from_pretrained(m['model_dir'], subfolder='transformer',
                    torch_dtype=torch.bfloat16, local_files_only=True).eval().to('cuda')
        structured, report['registration'] = register_wan(model)
        report['loaded_model_config'] = dict(model.config)
        report['resident_allocated_bytes'] = torch.cuda.memory_allocated()
        report['model_loads'] = dict(transformer=1, text_encoder=0, vae=0)
        install_compatibility(args, report)
        save(args.output, report)
        # Same OutputsError PTQ; only FLASH SDPA selection and data changed.
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ptq(structured, config.quant, cache=config.cache, load_dirpath='',
                save_dirpath=str(checkpoint), copy_on_save=False, save_model=True)
        torch.cuda.synchronize(); deadline(args)
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        files = [record(checkpoint / f'{name}.pt') for name in ('model', 'scale', 'wgts', 'branch', 'smooth')]
        identity = dict(experiment='E047', status='complete', model_dir=m['model_dir'],
                        manifest=record(args.manifest), files=files,
                        recipe='original-Wan real_nvfp4 + wan_s16, rank32, g10; matched 81-frame calibration',
                        collection=report['collection'], source=record(Path(__file__)))
        identity_path = Path(m['report_dir']) / 'checkpoint_identity.json'
        require(not identity_path.exists(), 'Preserve prior checkpoint receipt')
        save(identity_path, identity); report['checkpoint_identity'] = record(identity_path)
        save(checkpoint / 'artifact.json', dict(experiment='E047', model_dir=m['model_dir'],
             recipe=dict(format='wan-real-nvfp4-svdquant', rank=32, smooth_grids=10),
             calibration=report['collection'], checkpoint_identity=str(identity_path)))
        config.output.unlock(); report['config_output_directory'] = config.output.job_dirpath
    except BaseException:
        config.output.unlock(error=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['check', 'run'], required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args(); m = json.loads(args.manifest.read_text())
    args.output = args.output or Path(m['report_dir']) / ('ptq_check.json' if args.phase == 'check' else 'ptq_run.json')
    require(not args.output.exists(), 'Preserve prior reports')
    report = dict(experiment='E047', phase=args.phase, status='running', start_epoch=time.time(),
                  layer_events=[], sources=[record(Path(__file__)), record(args.manifest), record(PLAN)])
    try:
        os.environ['DEEPCOMPRESSOR_WAN_GATED'] = '0'
        # Match the successful historical import order; do not use run_module.
        import torch
        import pyarrow  # noqa: F401
        import diffusers
        from diffusers import WanTransformer3DModel
        require(args.phase != 'check' or not torch.cuda.is_available(), 'Hide CUDA during CPU check')
        report['environment'] = dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                                     diffusers=diffusers.__version__, cuda_available=torch.cuda.is_available())
        config, report['configuration'] = parse_config(m)
        from deepcompressor.app.diffusion.quant.utils import maybe_wan_eval_inputs, _wan_gated_enabled
        sentinel = object()
        require(not _wan_gated_enabled() and maybe_wan_eval_inputs(sentinel, None, '') is sentinel,
                'Disabled gating must preserve original evaluation inputs')
        model_config = WanTransformer3DModel.load_config(m['model_dir'], subfolder='transformer')
        with torch.device('meta'):
            meta = WanTransformer3DModel.from_config(model_config)
        _, report['meta_registration'] = register_wan(meta); del meta
        report['compatibility'] = dict(gated=False, gate_cache='local no-op', rotary='tuple preserved; move only if needed',
                                      model_loaded_for_check=False, collection_required_at_run=True)
        for rel in ('app/diffusion/ptq.py', 'app/diffusion/nn/struct.py', 'app/diffusion/dataset/calib.py',
                    'app/diffusion/quant/utils.py', 'app/diffusion/quant/smooth.py', 'app/diffusion/quant/weight.py'):
            report['sources'].append(record(DC / 'deepcompressor' / rel))
        if args.phase == 'run':
            require(args.deadline_unix is not None, 'Run requires an absolute deadline')
            run(args, m, config, report)
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc())
        raise
    finally:
        report['seconds'] = time.time()-report['start_epoch']; save(args.output, report)
        print(report['status'], args.output, flush=True)


if __name__ == '__main__':
    main()
