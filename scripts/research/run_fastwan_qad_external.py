#!/usr/bin/env python3
"""E024 official eager FastWan-QAD + FP16 TAEHV functional generation.

Only the runtime branch imports torch/FastVideo. Worker instrumentation observes
the official computation; synchronized stage timings are diagnostic, not a bench.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data1/models/svdquant-wjq')
REPORTS = ROOT / 'results/research/E024'
UPSTREAM = DATA / 'third_party/FastVideo-8444c089'
COMMIT = '8444c0897a8b96848eb85b6e5750ef486f79fc92'
MODEL = DATA / 'models/FastWan-QAD-1.3B-621c6aeb'
TAEHV = DATA / 'models/TAEHV-011dfc21'
ENTRY = 'examples/inference/optimizations/FastWan_QAD_TAEHV.py'
SAMPLING = dict(height=480, width=832, num_frames=81, fps=16,
                num_inference_steps=3, guidance_scale=1.0)


def file_record(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(file=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def configuration(args):
    manifest_path = ROOT / 'results/research/E022/video_test_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['trajectories']) == 16 and len(manifest['cases']) == 8
    trajectories = []
    for row in manifest['trajectories']:
        case = manifest['cases'][row['manifest_index']]
        assert row['prompt_id'] == case['prompt_id']
        assert row['seed'] == case['seeds'][row['replica']]
        trajectories.append(dict(row, prompt=case['prompt']))
    assert trajectories[0]['trajectory_id'] == 'vbench_128_r0'
    sources = [Path(__file__), UPSTREAM / ENTRY]
    sources += [UPSTREAM / name for name in (
        'fastvideo/api/schema.py', 'fastvideo/api/results.py',
        'fastvideo/pipelines/basic/wan/wan_pipeline.py',
        'fastvideo/models/schedulers/scheduling_flow_unipc_multistep.py',
        'fastvideo/layers/quantization/nvfp4_qat_config.py',
        'fastvideo/attention/backends/attn_qat_infer.py')]
    result = dict(experiment='E024', phase=args.phase, status='checking',
                  source={str(p): file_record(p) for p in sources},
                  input_manifest=file_record(manifest_path), sampling=SAMPLING,
                  trajectories=trajectories[:1] if args.phase == 'smoke' else trajectories,
                  build_args=dict(model=str(MODEL), distilled_model='', no_compile=True,
                                  baseline=False, num_gpus=1, taehv=True),
                  decoder=dict(class_name='official TaehvDecoder', dtype='torch.float16',
                               checkpoint=str(TAEHV / 'taew2_1.pth')),
                  limitations=[
                      'Functional eager run, no warmup, no performance or quality equivalence claim.',
                      'Same texts/seeds as E022, not the same noise/model/scheduler/frame count.',
                      'Official UniPC three-step inference differs from documented QAD DMD training schedule; no sampler substitution.',
                      'Stage logging and observation synchronizations make timings diagnostic.',
                      'Large asset SHA inherited from completed downloader manifest; this runner checks bytes/existence only.',
                      'Memory ceiling checked at stage boundaries; external supervisor enforces wall-clock deadline.'])
    result['upstream_commit'] = subprocess.check_output(
        ['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    assert result['upstream_commit'] == COMMIT
    missing = []
    candidates = ([args.asset_manifest] if args.asset_manifest else
                  sorted(REPORTS.glob('assets_manifest*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True))
    selected = next((p for p in candidates if p.exists() and
                     json.loads(p.read_text()).get('status') == 'complete'), None)
    if selected is None:
        missing.append('No completed asset manifest; no model import or model failure inferred.')
    else:
        assets = json.loads(selected.read_text())
        result['asset_manifest'] = file_record(selected)
        assert Path(assets['target']) == MODEL and Path(assets['taehv_target']) == TAEHV
        assert assets['hf_revision'] == '621c6aeb900f9f9a2ebb9ea9ed74c0daf31d5e6a'
        assert assets['taehv_revision'] == '011dfc2112197741c540e0bdd5b7b67bcc930771'
        for name, rec in assets['files'].items():
            p = Path(rec['path'])
            if rec.get('status') != 'verified' or not p.is_file() or p.stat().st_size != rec['bytes']:
                missing.append(name)
            elif p.stat().st_size <= 2 * 1024 * 1024 and file_record(p)['sha256'] != rec['sha256']:
                missing.append(name + ': small-file SHA mismatch')
        result['assets_verified_by_upstream'] = assets['files']
    configs = ['model_index.json', 'transformer/config.json', 'scheduler/scheduler_config.json',
               'text_encoder/config.json', 'vae/config.json', 'tokenizer/tokenizer_config.json']
    result['config_reference'] = {}
    for relative in configs:
        p = MODEL / relative
        if not p.is_file():
            missing.append(str(p))
        else:
            result['config_reference'][relative] = dict(file_record(p), content=json.loads(p.read_text()))
    for p in (MODEL / 'transformer/diffusion_pytorch_model.safetensors', TAEHV / 'taew2_1.pth', TAEHV / 'taehv.py'):
        if not p.is_file():
            missing.append(str(p))
    if 'model_index.json' in result['config_reference']:
        assert result['config_reference']['model_index.json']['content']['_class_name'] == 'WanPipeline'
    result.update(status='notready' if missing else 'complete', notready_reasons=missing,
                  gpu_imports_performed=False, torch_imported='torch' in sys.modules)
    assert not result['torch_imported']
    return result


def worker_observe(worker, action):
    """Serialized by value; installs only observation around existing callables."""
    import os
    import torch
    import fastvideo.layers.quantization.nvfp4_qat_config as qmod
    from fastvideo.attention.backends.attn_qat_infer import attn_qat_infer_receipt

    def small(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): small(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [small(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def memory():
        torch.cuda.synchronize()
        return {key: getattr(torch.cuda, func)() for key, func in (
            ('allocated', 'memory_allocated'), ('reserved', 'memory_reserved'),
            ('peak_allocated', 'max_memory_allocated'), ('peak_reserved', 'max_memory_reserved'))}

    model = worker.pipeline.get_module('transformer')
    scheduler = worker.pipeline.get_module('scheduler')
    if action == 'install':
        assert '_e024_monitor' not in worker.__dict__
        total_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        # RPC is available only after official construction. Retain its peak,
        # then cap subsequent worker PyTorch allocations (not non-PyTorch use).
        fraction = min(1.0, 60 * 2**30 / total_bytes)
        torch.cuda.set_per_process_memory_fraction(fraction)
        state = dict(handles=[], restores=[], linear_counts={}, attention_counts={},
                     dit_inputs=[], gemm_count=0, gemm_backends={}, linear_modules=[], attention_modules=[],
                     allocator_fraction=fraction, cap_installed_after_model_load=True,
                     model_load_memory_bytes=memory())
        worker._e024_monitor = state

        def dit_pre(module, inputs, kwargs):
            tensors = {str(i): v for i, v in enumerate(inputs) if isinstance(v, torch.Tensor)}
            tensors.update({k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
            state['dit_inputs'].append({k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in tensors.items()})

        state['handles'].append(model.register_forward_pre_hook(dit_pre, with_kwargs=True))
        seen = set()
        for name, mod in model.named_modules():
            if isinstance(getattr(mod, 'quant_method', None), qmod.NVFP4QATQuantizeMethod):
                assert 'weight' not in mod._parameters
                info = dict(name=name, dense_weight_present=getattr(mod, 'weight', None) is not None)
                for key in ('_fp4_weight', '_fp4_weight_scale', '_weight_global_sf'):
                    t = getattr(mod, key)
                    info[key] = dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device))
                state['linear_modules'].append(info)

                def linear_hook(module, inputs, output, key=name):
                    state['linear_counts'][key] = state['linear_counts'].get(key, 0) + 1
                state['handles'].append(mod.register_forward_hook(linear_hook))
            impl = getattr(mod, 'attn_impl', None)
            if impl is not None and id(impl) not in seen:
                seen.add(id(impl))
                kind = type(impl).__module__ + '.' + type(impl).__name__
                state['attention_modules'].append(dict(name=name, kind=kind, backend=str(getattr(mod, 'backend', None))))
                original = impl.forward

                def attention_forward(*a, _original=original, _key=kind, **kw):
                    state['attention_counts'][_key] = state['attention_counts'].get(_key, 0) + 1
                    return _original(*a, **kw)
                state['restores'].append((impl, 'forward', original))
                impl.forward = attention_forward
        original_mm = qmod._mm_fp4

        def counted_mm(*a, **kw):
            state['gemm_count'] += 1
            backend = str(kw.get('backend'))
            state['gemm_backends'][backend] = state['gemm_backends'].get(backend, 0) + 1
            return original_mm(*a, **kw)

        state['restores'].append((qmod, '_mm_fp4', original_mm))
        qmod._mm_fp4 = counted_mm
        assert len(state['linear_modules']) == 300 and len(state['attention_modules']) == 60
    state = worker._e024_monitor
    if action == 'reset':
        state['linear_counts'].clear()
        state['attention_counts'].clear()
        state['dit_inputs'].clear()
        state['gemm_count'] = 0
        state['gemm_backends'].clear()
        # Keep lifetime allocator peaks, including official model construction.
    if action == 'remove':
        for handle in state['handles']:
            handle.remove()
        for obj, attr, original in reversed(state['restores']):
            setattr(obj, attr, original)
        del worker._e024_monitor
        return {'removed': True}
    result = {key: small(value) for key, value in state.items() if key not in ('handles', 'restores')}
    result.update(pid=os.getpid(), memory_bytes=memory(), backend_receipt=attn_qat_infer_receipt(),
                  cuda_device=torch.cuda.get_device_name(), cuda_capability=list(torch.cuda.get_device_capability()),
                  scheduler=dict(class_name=type(scheduler).__module__ + '.' + type(scheduler).__name__,
                                 config=small(dict(scheduler.config)),
                                 timesteps=small(getattr(scheduler, 'timesteps', None)),
                                 sigmas=small(getattr(scheduler, 'sigmas', None))))
    return result


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(__import__('torch').uint8).numpy()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(raw).hexdigest(), finite=bool(value.isfinite().all()))


def generate(args, report, output):
    # Set before any torch/FastVideo import; no implicit network fallback.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      FASTVIDEO_ATTENTION_BACKEND='ATTN_QAT_INFER', FASTVIDEO_STAGE_LOGGING='1')
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(TAEHV))
    import cloudpickle
    import imageio
    import torch

    spec = importlib.util.spec_from_file_location('e024_official_entry', UPSTREAM / ENTRY)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    import taehv
    assert file_record(taehv.__file__)['sha256'] == file_record(TAEHV / 'taehv.py')['sha256']
    report.update(status='running', started_unix=time.time(), pid=os.getpid(),
                  deadline_unix=args.deadline_unix, cases=[], versions=dict(torch=torch.__version__),
                  taehv_runtime_source=file_record(taehv.__file__))
    write_json(output, report)
    artifact_dir = args.artifact_dir or DATA / 'research/20261003/E024/videos' / args.phase
    artifact_dir.mkdir(parents=True, exist_ok=False)
    generator = None
    rpc_installed = False
    started = time.perf_counter()

    def deadline():
        if time.time() >= args.deadline_unix:
            raise TimeoutError('E024 supplied wall-clock deadline exceeded')

    def rpc(action):
        receipts = generator.executor.collective_rpc(cloudpickle.dumps(worker_observe), args=(action,))
        assert len(receipts) == 1
        return receipts[0]

    def memory_guard(receipt):
        torch.cuda.synchronize()
        parent = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                      peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        # Separate allocator processes; the sum of lifetime peaks is conservative.
        combined = parent['peak_allocated'] + receipt['memory_bytes']['peak_allocated']
        result = dict(parent_bytes=parent, worker_bytes=receipt['memory_bytes'],
                      sum_lifetime_peak_allocated_gib=combined / 2**30)
        if combined > 60 * 2**30:
            report['memory_ceiling_failure'] = result
            raise MemoryError('Observed sum of parent/worker allocator peaks exceeds 60 GiB')
        return result

    try:
        deadline()
        t0 = time.perf_counter()
        generator = official.build_generator(argparse.Namespace(**report['build_args']))
        report['build_seconds'] = time.perf_counter() - t0
        report['worker_installation'] = rpc('install')
        rpc_installed = True
        assert report['worker_installation']['cuda_capability'] == [12, 0]
        assert 'kernel=fastvideo-kernel-cutlass' in report['worker_installation']['backend_receipt']
        report['build_memory'] = memory_guard(report['worker_installation'])
        deadline()
        decoder = official.TaehvDecoder(str(TAEHV / 'taew2_1.pth'), device='cuda', dtype=torch.float16)
        decoded_info = {}
        original_decode = decoder.model.decode_video

        def observed_decode(*a, **kw):
            value = original_decode(*a, **kw)
            decoded_info.update(shape=list(value.shape), dtype=str(value.dtype), finite=bool(torch.isfinite(value).all()))
            if not decoded_info['finite']:
                raise FloatingPointError('Nonfinite TAEHV floating output before clamp/uint8 conversion')
            return value

        decoder.model.decode_video = observed_decode
        write_json(output, report)
        for trajectory in report['trajectories']:
            deadline()
            row = dict(trajectory, status='running')
            report['cases'].append(row)
            write_json(output, report)
            tid = trajectory['trajectory_id']
            video_path = artifact_dir / (tid + '.mp4')
            request = dict(prompt=trajectory['prompt'], sampling=dict(SAMPLING, seed=trajectory['seed']),
                           runtime=dict(return_trajectory_latents=True),
                           output=dict(save_video=False, return_frames=True, output_path=str(video_path)))
            row['request'] = request
            rpc('reset')
            t0 = time.perf_counter()
            result = generator.generate(request=request)
            row['generate_wall_seconds'] = time.perf_counter() - t0
            row['generation_time_pipeline_seconds'] = result.generation_time
            row['official_logging_info'] = getattr(result.logging_info, 'stages', result.logging_info)
            row['official_e2e_latency_seconds'] = result.extra.get('e2e_latency')
            row['backend_receipt'] = receipt = rpc('read')
            row['memory'] = memory_guard(receipt)
            tensors = dict(final_latents=result.samples, trajectory=result.trajectory,
                           trajectory_timesteps=result.trajectory_timesteps)
            tensor_path = artifact_dir / (tid + '_latents.pt')
            torch.save(tensors, tensor_path)
            row['latents_artifact'] = file_record(tensor_path)
            row['tensors'] = {key: tensor_record(value) for key, value in tensors.items()}
            write_json(output, report)
            assert all(item['finite'] for item in row['tensors'].values())
            assert list(result.samples.shape) == [1, 16, 21, 60, 104]
            assert list(result.trajectory.shape) == [1, 3, 16, 21, 60, 104]
            assert torch.equal(result.samples.float(), result.trajectory[:, -1].float())
            row['actual_timesteps'] = result.trajectory_timesteps.tolist()
            assert row['actual_timesteps'] == receipt['scheduler']['timesteps']
            assert receipt['scheduler']['class_name'].endswith('FlowUniPCMultistepScheduler')
            assert receipt['scheduler']['config']['shift'] == 3
            assert len(receipt['dit_inputs']) == 3
            assert len(receipt['linear_counts']) == 300 and set(receipt['linear_counts'].values()) == {3}
            assert receipt['gemm_count'] == 900 and receipt['gemm_backends'] == {'cutlass': 900}
            attention = receipt['attention_counts']
            fp4 = sum(n for k, n in attention.items() if k.endswith('.AttnQatInferImpl'))
            dense = sum(n for k, n in attention.items() if k.endswith(('.FlashAttentionImpl', '.SDPAImpl')))
            assert fp4 == 90 and dense == 90 and sum(attention.values()) == 180
            row['actual_calls'] = dict(dit=3, nvfp4_gemm=900, fp4_attention=fp4, dense_cross_attention=dense)
            deadline()
            decoded_info.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            frames = decoder.decode(result.samples)
            torch.cuda.synchronize()
            row['taehv_decode_seconds'] = time.perf_counter() - t0
            row['taehv_floating_output'] = dict(decoded_info)
            row['frames_shape'] = list(frames.shape)
            row['frames_sha256'] = hashlib.sha256(frames.tobytes()).hexdigest()
            assert list(frames.shape) == [81, 480, 832, 3] and str(frames.dtype) == 'uint8'
            row['memory'] = memory_guard(receipt)
            t0 = time.perf_counter()
            imageio.mimsave(str(video_path), frames, fps=16, format='mp4')
            row['video_write_seconds'] = time.perf_counter() - t0
            row['video'] = file_record(video_path)
            row['status'] = 'complete'
            write_json(artifact_dir / (tid + '.json'), row)
            write_json(output, report)
            print(json.dumps(dict(trajectory_id=tid, status=row['status'], calls=row['actual_calls'],
                                  generate_seconds=row['generate_wall_seconds']), ensure_ascii=False), flush=True)
            del result, tensors, frames
        deadline()
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed_partial_preserved', error=traceback.format_exc())
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            report['cases'][-1]['status'] = 'failed'
        raise
    finally:
        if generator is not None:
            try:
                if rpc_installed:
                    report['observation_cleanup'] = rpc('remove')
            except BaseException:
                report['cleanup_error'] = traceback.format_exc()
            try:
                generator.shutdown()
            except BaseException:
                report['shutdown_error'] = traceback.format_exc()
                report['status'] = 'failed_partial_preserved'
        report.update(ended_unix=time.time(), wall_seconds=time.perf_counter() - started)
        write_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['smoke', 'suite'], required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--asset-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--artifact-dir', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    suffix = 'cpucheck' if args.check_only else 'run'
    output = args.output or REPORTS / f'{args.phase}_{suffix}.json'
    if output.exists():
        raise FileExistsError(f'Preserving prior result: {output}; use --output for a new attempt')
    report = configuration(args)
    if args.check_only:
        write_json(output, report)
        print(json.dumps(dict(status=report['status'], output=str(output),
                              notready_reasons=report['notready_reasons']), ensure_ascii=False))
        return
    if report['status'] != 'complete':
        write_json(output, report)
        raise RuntimeError('Local assets are not ready; no model imported')
    if args.deadline_unix is None or not 0 < args.deadline_unix - time.time() <= 1200:
        raise ValueError('Supply supervisor --deadline-unix, at most 1200 seconds in the future')
    try:
        generate(args, report, output)
    except BaseException:
        # Also preserve import/construction errors before generate's inner try.
        report.update(status='failed_partial_preserved', error=traceback.format_exc())
        write_json(output, report)
        raise


if __name__ == '__main__':
    main()
