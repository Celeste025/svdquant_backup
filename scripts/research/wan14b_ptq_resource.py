#!/usr/bin/env python3
"""E052: full-budget Wan14B block0 smoothing/LR resource pilot, not a checkpoint."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

from ptq_wan_matched_calibration import ROOT, DC, EXAMPLES, require, record, save

MANIFEST = ROOT / 'research_state/06_experiments/E052_wan14b_ptq_resource_manifest.json'
PLAN = ROOT / 'research_state/06_experiments/E052_wan14b_ptq_resource_plan.md'


def deadline(args):
    if args.deadline_unix is not None and time.time() >= args.deadline_unix:
        raise TimeoutError('E052 resource-pilot deadline reached; no reduced-budget retry')


def checked_record(rec):
    path = Path(rec['file'])
    actual = record(path)
    require(all(actual[k] == rec[k] for k in ('bytes', 'sha256')), f'Input binding changed: {path}')
    return path


def parse_config(m):
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    import torch
    files = ['configs/model/wan2.1-1.3b.yaml', 'configs/svdquant/real_nvfp4.yaml',
             'configs/svdquant/wan_s16.yaml']
    s, p = m['settings'], m['ptq']
    argv = [*files, '--pipeline-name=wan2.1-14b', '--pipeline-path=' + m['model_dir'],
            '--calib-path=' + m['calibration_path'], '--calib-num-samples=64',
            '--output-root=' + str(Path(m['data_dir']) / 'config_output'),
            '--cache-root=' + str(Path(m['data_dir']) / 'config_cache'),
            '--skip-eval', '--skip-gen', '--eval-num-gpus=1', '--eval-batch-size-per-gpu=1',
            '--eval-num-frames=81', '--eval-num-steps=50', '--eval-guidance-scale=5.0',
            '--eval-height=480', '--eval-width=832', '--eval-protocol=unipc50-g5.0-f81-shift3',
            '--seed=' + str(m['random_seed'])]
    cwd, old_argv = Path.cwd(), sys.argv
    try:
        os.chdir(EXAMPLES); sys.argv = ['E052-resource', *argv]
        config, _, unused_configs, unused_args, unknown = DiffusionPtqRunConfig.get_parser().parse_known_args()
        require(not unknown and not unused_args, f'Unparsed configuration: {unknown}, {unused_args}')
    finally:
        os.chdir(cwd); sys.argv = old_argv
    q = config.quant
    require(config.pipeline.name == 'wan2.1-14b' and config.pipeline.path == m['model_dir'], '14B identity')
    require((s['height'], s['width'], s['num_frames'], s['num_inference_steps'],
             s['guidance_scale'], s['flow_shift']) == (480, 832, 81, 50, 5., 3.), 'E050 sampling contract')
    require((q.calib.num_samples, q.calib.batch_size) ==
            (p['num_samples'], p['calibration_batch_size']) == (64, 4), 'Full64/batch4 required')
    require((q.wgts.low_rank.rank, q.wgts.low_rank.num_iters, q.wgts.low_rank.early_stop) ==
            (p['rank'], p['low_rank_max_iters'], p['low_rank_early_stop']) == (32, 100, True), 'LR recipe')
    require(q.smooth.proj.num_grids == p['smooth_grids'] == 10, 'Full smoothing grid')
    for search in (q.smooth.proj, q.wgts.low_rank):
        require(search.sample_size == p['sample_size'] == -1 and
                search.sample_batch_size == p['sample_batch_size'] == 4, 'All samples, batch4')
        require(search.objective.name == 'OutputsError', 'OutputsError objective required')
    require(not config.pipeline.shift_activations and not q.ipts.static and
            not p['shift_activations'] and not p['gated_output_error'], 'Dynamic activation/ungated recipe')
    for quant in (q.wgts, q.ipts):
        require(str(quant.dtype) == p['weight_activation_dtype'] == 'sfp4_e2m1_all', 'W/A E2M1')
        require(quant.scale_dtypes[0] == torch.float32 and
                str(quant.scale_dtypes[1]) == 'sfp8_e4m3_nan', 'FP32 global/E4M3 local scales')
        require(quant.group_shapes[-1][1] == p['group_size'] == 16, 'Group16')
    return config, dict(argv=argv, unused_configs=unused_configs, unused_args=unused_args,
        config_files=[record(EXAMPLES / p) for p in files], resolved_config=config.dump(),
        inherited_yaml_note='Reuse numerical Wan recipe; pipeline name/path,81f,CFG5 and shift3 protocol overridden. '
                            'No generation or scheduler executes in this pilot.')


def register_wan(model):
    import torch
    from deepcompressor.app.diffusion.nn.struct import DiffusionAttentionStruct, DiffusionModelStruct
    import deepcompressor.app.diffusion.dataset.calib as calib
    kind = type(model.blocks[0].attn1)
    if kind not in DiffusionAttentionStruct._factories:
        DiffusionAttentionStruct.register_factory(kind, DiffusionAttentionStruct._default_construct)
    if kind not in calib._ATTN_TYPES:
        calib._ATTN_TYPES = (*calib._ATTN_TYPES, kind)
    structured = DiffusionModelStruct.construct(model)
    targets = [name for name, mod in model.named_modules()
               if name.startswith('blocks.') and isinstance(mod, torch.nn.Linear)]
    first = [name for name in targets if name.startswith('blocks.0.')]
    require(structured.module is model and len(model.blocks) == 40 and len(targets) == 400 and len(first) == 10,
            'Expected real14B40blocks/400linears/block0ten')
    require(model.config.num_attention_heads == 40 and model.config.ffn_dim == 13824, '14B architecture')
    return structured, dict(blocks=40, target_count=400, block0_targets=first,
        attention_type=f'{kind.__module__}.{kind.__name__}', direct_transformer=True)


def collection_ready(m):
    cm = json.loads(checked_record(m['collection_manifest']).read_text())
    require(cm['experiment'] == 'E050' and cm['model_dir'] == m['model_dir'] and
            cm['settings'] == m['settings'] and cm['selected_caches'] == m['selected_caches'], 'E050 identity')
    rd = Path(m['collection_dir'])
    launcher_path, summary_path = rd / 'collect_launcher.json', rd / 'collection_summary.json'
    launcher, summary = json.loads(launcher_path.read_text()), json.loads(summary_path.read_text())
    require(launcher['status'] == summary['status'] == 'complete' and
            launcher['experiment'] == summary['experiment'] == 'E050', 'E050 collection/audit incomplete')
    require(launcher['manifest'] == summary['current_manifest'] == m['collection_manifest'], 'E050 manifest binding')
    require(all(w['status'] == 'complete' and w['returncode'] == 0 for w in launcher['workers']), 'E050 workers')
    require(summary['cache_count'] == 64 and summary['prompt_count'] == 14, 'Complete E05064/14')
    expected = {r['filename'] for r in m['selected_caches']}
    cache_dir = Path(m['calibration_path']) / 'caches'
    require(len(expected) == 64 and {p.name for p in cache_dir.glob('*.pt')} == expected, 'Exact64cache set')
    rows = summary['caches']
    require({r['filename'] for r in rows} == expected and len(rows) == 64, 'Independent audit coverage')
    for r in rows:
        path = cache_dir / r['filename']
        require(path.resolve() == Path(r['artifact']['file']).resolve() and
                path.stat().st_size == r['artifact']['bytes'], 'Audited cache file/path/size')
    return dict(manifest=m['collection_manifest'], launcher=record(launcher_path), summary=record(summary_path),
        caches=[r['artifact'] for r in rows], cache_hash_policy='Hashes inherited from complete independent E050 audit; '
        'exact directory membership/path/bytes checked here, not a second tensor-content audit.')


def memory():
    import torch
    import psutil
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
        cpu_rss_bytes=psutil.Process().memory_info().rss,
        cpu_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


@contextmanager
def stage(name, args, report, counters):
    import torch
    import psutil
    deadline(args); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    row = dict(stage=name, status='running', before=memory(), counts_before=dict(counters))
    report['stages'].append(row); report['active_stage'] = name; save(args.output, report)
    stop = threading.Event(); samples = [row['before']['cpu_rss_bytes']]
    process = psutil.Process()
    def sample_rss():
        while not stop.wait(.25):
            samples.append(process.memory_info().rss)
    sampler = threading.Thread(target=sample_rss, daemon=True); sampler.start()
    started = time.monotonic()
    try:
        yield row
        torch.cuda.synchronize(); deadline(args); row['status'] = 'complete'
    except BaseException:
        row['status'] = 'failed_preserved'
        raise
    finally:
        stop.set(); sampler.join()
        row.update(synchronized_wall_seconds=time.monotonic()-started, after=memory(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cpu_sampled_peak_rss_bytes=max(samples + [process.memory_info().rss]),
            cpu_rss_sampling_seconds=.25,
            actual_counts={k: counters[k] - row['counts_before'].get(k, 0) for k in counters})
        row['wall_note'] = 'Successful stage ends with CUDA synchronization; failed stage may have incomplete work.'
        report['actual_counts'] = dict(counters); save(args.output, report)
        print('E052 stage', name, row['status'], row['synchronized_wall_seconds'], flush=True)


@contextmanager
def instrumentation(model, args, report, counters):
    from deepcompressor.calib.search import SearchBasedCalibrator
    from deepcompressor.calib.smooth import SmoothCalibrator
    from deepcompressor.calib.lowrank import QuantLowRankCalibrator
    original = {name: getattr(SearchBasedCalibrator, name) for name in ('calibrate', 'ask', 'tell')}
    signature = inspect.signature(original['calibrate'])
    weights = {id(p): name for name, p in model.named_parameters()}
    active = {}
    hooks = []
    def count(key):
        def hook(*unused):
            deadline(args); counters[key] += 1
        return hook
    hooks.append(model.register_forward_pre_hook(count('transformer_prefix_entries')))
    for i, block in enumerate(model.blocks):
        counters[f'block_{i:02d}_forwards'] = 0
        hooks.append(block.register_forward_pre_hook(count(f'block_{i:02d}_forwards')))
    for name in report['registration']['block0_targets']:
        counters[name] = 0
        hooks.append(model.get_submodule(name).register_forward_pre_hook(count(name)))
    def calibrate(self, *a, **kw):
        if not isinstance(self, (SmoothCalibrator, QuantLowRankCalibrator)):
            return original['calibrate'](self, *a, **kw)
        deadline(args)
        bound = signature.bind_partial(self, *a, **kw).arguments
        ipts = bound.get('eval_inputs')
        row = dict(index=len(report['searches']), stage=report['active_stage'],
            calibrator=type(self).__name__, status='running',
            weight_names=[weights.get(id(w), '<nonparameter>') for w in bound.get('x_wgts') or []],
            sample_size=self.config.sample_size, sample_batch_size=self.config.sample_batch_size,
            input_num_samples=ipts.front().num_samples if ipts is not None else None,
            max_iterations=self.num_iters, ask_count=0, tell_count=0)
        report['searches'].append(row); active[id(self)] = row; save(args.output, report)
        started = time.monotonic()
        try:
            result = original['calibrate'](self, *a, **kw)
            row.update(status='complete', iterations_completed=self.iter,
                early_stopped=bool(getattr(self, 'early_stopped', False)), population_size=self.population_size)
            return result
        except BaseException:
            row['status'] = 'failed_preserved'; raise
        finally:
            row['wall_seconds_including_original_search'] = time.monotonic()-started
            active.pop(id(self), None); save(args.output, report)
    def ask(self):
        deadline(args)
        result = original['ask'](self)
        if id(self) in active:
            active[id(self)]['ask_count'] += 1
        return result
    def tell(self, error):
        result = original['tell'](self, error)
        if id(self) in active:
            row = active[id(self)]; row['tell_count'] += 1
            row['early_stopped'] = bool(getattr(self, 'early_stopped', False))
        return result
    SearchBasedCalibrator.calibrate, SearchBasedCalibrator.ask, SearchBasedCalibrator.tell = calibrate, ask, tell
    try:
        yield
    finally:
        for name, method in original.items():
            setattr(SearchBasedCalibrator, name, method)
        for hook in hooks:
            hook.remove()


@contextmanager
def first_block_only(structured):
    import deepcompressor.app.diffusion.dataset.calib as calib
    from deepcompressor.app.diffusion.quant.utils import _wan_gated_enabled
    require(not _wan_gated_enabled(), 'Historical gated calibration must remain disabled')
    original_args = structured.get_iter_layer_activations_args
    original_attach = calib._attach_wan_gate_eval_kwargs
    def limited(**kwargs):
        values = original_args(**kwargs)
        require(len(values) == 4 and values[0][0] is structured.module.blocks[0], 'First real block')
        require(values[2][0] is False and values[3][0] is False, 'Original block0 recompute/use-prev contract')
        return tuple(value[:1] for value in values)
    def no_gate(*a, **kw):
        require(not _wan_gated_enabled(), 'Unexpected gated replay')
    structured.get_iter_layer_activations_args = limited
    calib._attach_wan_gate_eval_kwargs = no_gate
    try:
        yield
    finally:
        structured.get_iter_layer_activations_args = original_args
        calib._attach_wan_gate_eval_kwargs = original_attach


def hook_signature(model):
    hooks = {name: {kind: list(getattr(mod, kind))
                   for kind in ('_forward_pre_hooks', '_forward_hooks')}
             for name, mod in model.named_modules()}
    return dict(sha256=hashlib.sha256(json.dumps(hooks, sort_keys=True).encode()).hexdigest(),
                hook_count=sum(len(ids) for item in hooks.values() for ids in item.values()))


def acquire_cache(structured, config, report, counters):
    import torch
    from deepcompressor.app.diffusion.quant.utils import get_needs_inputs_fn
    hooks_before, counts_before = hook_signature(structured.module), dict(counters)
    loader = config.quant.calib.build_loader()
    require(len(loader.dataset) == 64 and loader.batch_size == 4, 'Full data loader budget')
    gen = loader.iter_layer_activations(structured,
        needs_inputs_fn=get_needs_inputs_fn(structured, config.quant),
        skip_pre_modules=True, skip_post_modules=True, early_stop_module=structured.module.blocks[0])
    name, (layer, cache, kwargs) = next(gen)
    require(name == 'blocks.0' and layer.module is structured.module.blocks[0], 'Only block0 cache')
    rot = kwargs.get('rotary_emb')
    require(isinstance(rot, tuple) and len(rot) == 2 and all(isinstance(t, torch.Tensor) for t in rot),
            'Diffusers0.40 cosine/sine tuple required')
    device = next(layer.module.parameters()).device
    require(rot[0].shape == rot[1].shape and rot[0].shape[0] == 1 and rot[0].shape[2] == 1,
            'Original E047 rotary broadcasting contract')
    kwargs['rotary_emb'] = tuple(t.to(device=device) if t.device != device else t for t in rot)
    cache_info = {}
    for key, io in cache.items():
        if io.inputs is None:
            continue
        cache_info[key] = {}
        for tensor_key, front in io.inputs.tensors.items():
            actual_rows = sum(t.shape[0] for t in front.data)
            require(front.num_samples == front.num_total == front.num_cached == actual_rows == 64,
                    f'Incomplete actual cache rows in {key}/{tensor_key}')
            cache_info[key][str(tensor_key)] = dict(num_samples=front.num_samples,
                num_total=front.num_total, num_cached=front.num_cached, actual_rows=actual_rows,
                data_batches=len(front.data), shapes=[list(t.shape) for t in front.data],
                dtype=str(front.data[0].dtype), cache_device=str(front.data[0].device),
                orig_device=str(front.orig_device))
    hooks_after = hook_signature(structured.module)
    require(hooks_before == hooks_after, 'Cache collection left temporary model hooks installed')
    actual_counts = {k: counters[k] - counts_before.get(k, 0) for k in counters}
    require(actual_counts['transformer_prefix_entries'] == 16 and actual_counts['block_00_forwards'] == 32 and
            all(actual_counts[f'block_{i:02d}_forwards'] == 0 for i in range(1, 40)), 'Actual16prefix/32block0/0downstream')
    report['cache_passes'].append(dict(stage=report['active_stage'], layer=name, modules=cache_info,
        rotary_shapes=[list(t.shape) for t in rot], rotary_dtype=[str(t.dtype) for t in rot],
        records=64, batch_size=4, fresh_loader=True, only_block0=True, actual_counts=actual_counts,
        hooks_before=hooks_before, hooks_after=hooks_after,
        hook_note='Each pass captures its own baseline; the second includes legitimate smoothing hooks.'))
    return loader, gen, layer, cache, kwargs


def release_cache(gen, cache):
    # Exhaust the one-layer iterator to execute its original cache-clearing path.
    require(next(gen, None) is None, 'Unexpected second block was exposed')
    gen.close()
    for io in cache.values():
        io.clear()


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    return value


def run(args, m, config, report):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import WanTransformer3DModel
    from deepcompressor.app.diffusion.quant.smooth import smooth_diffusion_layer
    from deepcompressor.app.diffusion.quant.weight import calibrate_diffusion_block_low_rank_branch
    from deepcompressor.utils import tools
    deadline(args); report['collection'] = collection_ready(m)
    data = Path(m['data_dir']); data.mkdir(parents=True, exist_ok=True)
    for name in ('partial_smooth.pt', 'partial_branch.pt'):
        require(not (data / name).exists(), 'Preserve earlier partial artifacts')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tools.logging.setup(path=str(Path(m['report_dir']) / 'calibration.log'), level=tools.logging.DEBUG)
    config.dump(path=str(data / 'resolved_config.yaml'))
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    counters = {'transformer_prefix_entries': 0}
    with stage('model_load', args, report, counters):
        model = WanTransformer3DModel.from_pretrained(m['model_dir'], subfolder='transformer',
            torch_dtype=torch.bfloat16, local_files_only=True).eval().to('cuda')
        structured, report['registration'] = register_wan(model)
        report['loaded_model_config'] = dict(model.config)
        require(all(p.device.type == 'cuda' for p in model.parameters()), 'All14B parameters must remain resident')
        report['resident_parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
        report['model_loads'] = dict(transformer=1, text_encoder=0, vae=0)
        report['model_weights_loaded'] = True
    torch.manual_seed(m['random_seed'])
    report['calibration_random_seed'] = dict(seed=m['random_seed'], reset_after_model_load=True,
        interpretation='Fix this pilot randomized-SVD stream, not the full-model phase ordering.')
    smooth, branches = {}, {}
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
            first_block_only(structured), instrumentation(model, args, report, counters):
        with stage('smooth_cache', args, report, counters):
            loader, gen, layer, cache, kwargs = acquire_cache(structured, config, report, counters)
        with stage('smooth', args, report, counters):
            smooth_diffusion_layer(layer=layer, config=config.quant, smooth_cache=smooth,
                                   layer_cache=cache, layer_kwargs=kwargs)
        smooth['proj.fuse_when_possible'] = config.quant.smooth.proj.fuse_when_possible
        torch.save(cpu_copy(smooth), data / 'partial_smooth.pt')
        report['partial_smooth'] = dict(artifact=record(data / 'partial_smooth.pt'), keys=list(smooth),
            scope='blocks.0 only; not a complete smooth.pt and not accepted by full PTQ load_dirpath')
        release_cache(gen, cache); del loader, gen, layer, cache, kwargs
        gc.collect(); torch.cuda.empty_cache(); save(args.output, report)
        with stage('lr_cache', args, report, counters):
            loader, gen, layer, cache, kwargs = acquire_cache(structured, config, report, counters)
        with stage('low_rank', args, report, counters):
            calibrate_diffusion_block_low_rank_branch(layer=layer, config=config.quant,
                branch_state_dict=branches, layer_cache=cache, layer_kwargs=kwargs)
        torch.save(cpu_copy(branches), data / 'partial_branch.pt')
        report['partial_branch'] = dict(artifact=record(data / 'partial_branch.pt'), keys=list(branches),
            scope='blocks.0 only; weights already smoothed during calibration; no full model export')
        release_cache(gen, cache); del loader, gen, layer, cache, kwargs
        require(sum(counters[f'block_{i:02d}_forwards'] for i in range(1, 40)) == 0, 'Downstream block executed')
        targets = set(report['registration']['block0_targets'])
        for stage_name in ('smooth', 'low_rank'):
            rows = [r for r in report['searches'] if r['stage'] == stage_name]
            covered = {name.removesuffix('.weight') for r in rows for name in r['weight_names']}
            require(targets <= covered and all(r['status'] == 'complete' for r in rows), 'Incomplete10linear search coverage')
        report['coverage'] = dict(smoothing_main_linears=10, low_rank_main_linears=10,
            downstream_block_forwards=0, cache_passes=2, records_per_pass=64,
            all_parameters_still_cuda=all(p.device.type == 'cuda' for p in model.parameters()))
    report['actual_counts'] = dict(counters)
    report['final_memory'] = memory()
    report['partial_artifact_note'] = ('No full model/weight-range/activation calibration, export or quality evaluation. '
        'Any later partial-cache reuse requires explicit per-layer dispatch; stock full-PTQ treats nonempty cache as complete. '
        'Pilot seed fixes randomized SVD here, not whole-model RNG ordering.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'run'), required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args(); m = json.loads(args.manifest.read_text())
    args.output = args.output or Path(m['report_dir']) / ('check.json' if args.phase == 'check' else 'run.json')
    require(not args.output.exists(), 'Preserve previous report; select another output explicitly')
    report = dict(experiment='E052', phase=args.phase, status='running', start_epoch=time.time(),
        pid=os.getpid(), manifest=record(args.manifest), source=record(Path(__file__)), plan=record(PLAN),
        model_weights_loaded=False, stages=[], searches=[], cache_passes=[])
    report['sources'] = [report['source'], report['manifest'], report['plan'],
                        record(ROOT / 'scripts/research/ptq_wan_matched_calibration.py')]
    try:
        require(m['experiment'] == 'E052', 'Wrong experiment')
        os.environ['DEEPCOMPRESSOR_WAN_GATED'] = '0'
        import torch
        import pyarrow  # noqa: F401; retain successful historical import order
        import diffusers
        from diffusers import WanTransformer3DModel
        torch.set_num_threads(4)
        if args.phase == 'check':
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available(), 'Hide CUDA for check')
        report['environment'] = dict(python=sys.executable, torch=torch.__version__, diffusers=diffusers.__version__,
            cuda=torch.version.cuda, cuda_initialized_before=torch.cuda.is_initialized())
        cm = json.loads(checked_record(m['collection_manifest']).read_text())
        require(cm['model_dir'] == m['model_dir'] and cm['settings'] == m['settings'] and
                cm['selected_caches'] == m['selected_caches'], 'Bound14B collection configuration')
        config, report['configuration'] = parse_config(m)
        model_config = WanTransformer3DModel.load_config(m['model_dir'], subfolder='transformer', local_files_only=True)
        with torch.device('meta'):
            meta = WanTransformer3DModel.from_config(model_config)
        structured, report['meta_registration'] = register_wan(meta)
        with first_block_only(structured):
            values = structured.get_iter_layer_activations_args(skip_pre_modules=True, skip_post_modules=True)
            require(all(len(v) == 1 for v in values) and values[0][0] is meta.blocks[0] and
                    values[2] == [False] and values[3] == [False], 'One-layer transport geometry')
        del meta, structured
        for rel in ('app/diffusion/dataset/calib.py', 'dataset/cache.py', 'app/diffusion/nn/struct.py',
                    'app/diffusion/quant/smooth.py', 'app/diffusion/quant/weight.py', 'calib/search.py',
                    'calib/smooth.py', 'calib/lowrank.py', 'nn/patch/lowrank.py'):
            report['sources'].append(record(DC / 'deepcompressor' / rel))
        report['compatibility'] = dict(gated=False, rotary='original cosine/sine tuple retained and moved to block device',
            collection_required_only_at_run=True, check_loads_weights=False, block0_only=True)
        if args.phase == 'run':
            require(args.deadline_unix is not None, 'Run requires explicit deadline')
            run(args, m, config, report)
        else:
            require(not torch.cuda.is_initialized(), 'CUDA initialized in check')
            report['cuda_initialized'] = False
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc()); raise
    finally:
        report['seconds'] = time.time() - report['start_epoch']; save(args.output, report)
        print(report['status'], args.output, flush=True)


if __name__ == '__main__':
    main()
