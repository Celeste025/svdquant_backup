#!/usr/bin/env python3
"""E038 v2: restrict the no-CUDA-initialization assertion to CPU checks.

Original H3 sampler/step semantics; actual prepared noise is injected, not
regenerated from the seed. Only scalar step receipts and final latents are saved.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_h3_native_paired_video as old
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
from probe_h3_plain_baseline import file_record, tensor_record, tree_signature
from h3_coarse16_attention import ARMS, cpu_geometry_check, install_h3_center_attention

MANIFEST = ROOT/'research_state/06_experiments/E038_center_video_manifest.json'
MODALITIES = ('video', 'audio')
save = old.save


def require(value, message):
    if not value:
        raise RuntimeError(message)


def schedules(pipe, cfg):
    pipe.scheduler.set_timesteps(cfg['num_inference_steps'], shift=cfg['flow_shift'])
    pipe.scheduler_audio.set_timesteps(cfg['num_inference_steps'], shift=cfg['audio_flow_shift'])
    return actual_schedules(pipe)


def actual_schedules(pipe):
    return {m: dict(timesteps=s.timesteps.tolist(), sigmas=s.sigmas.tolist())
        for m, s in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}


def budget(args, report, before_forward=False):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'Original worker deadline expired or absent')
    if before_forward:
        require(report['attempted_dit_calls'] < report['allocated_dit_calls'], 'DiT allocation exhausted')
    old.memory_guard()


def tensor_scalars(value):
    x = value.detach().float()
    finite = bool(torch.isfinite(x).all())
    require(finite, 'Nonfinite recurrent tensor')
    return dict(finite=True, rms=float(x.square().mean().sqrt()), max_abs=float(x.abs().max()))


def read_manifest(args, report):
    manifest = json.loads(args.manifest.read_text())
    require(manifest['experiment'] == 'E038' and manifest['arms'] == list(ARMS), 'Wrong E038 protocol')
    require(len(manifest['cases']) == 8 and len({r['case_id'] for r in manifest['cases']}) == 8,
            'Expected eight unique fixed trajectories')
    cfg = manifest['settings']
    require((cfg['height'], cfg['width'], cfg['num_frames'], cfg['num_inference_steps'],
             cfg['cfg_scale'], cfg['flow_shift'], cfg['audio_flow_shift']) == (576,1024,124,20,1.,12.,3.),
            'Original E017 sampling contract changed')
    require(cfg['rand_device'] == 'cpu' and cfg['noise_dtype'] == 'bfloat16', 'Noise contract changed')
    args.data_dir = args.data_dir or Path(manifest['data_dir'])
    args.report_dir = args.report_dir or Path(manifest['report_dir'])
    args.prepare_report = args.prepare_report or args.report_dir/'prepare.json'
    args.output = args.output or args.report_dir/f'{args.phase}_{args.arm}.json'
    require(not args.output.exists(), f'Preserve prior report: {args.output}')
    args.report_writable = True
    require(args.data_dir == Path(manifest['data_dir']) and args.report_dir == Path(manifest['report_dir']),
            'Data/report paths differ from manifest')
    export = Path(manifest['export_dir'])/'manifest.json'
    e009 = json.loads((ROOT/'results/research/E009_h3_full.json').read_text())
    require(old.sha256(export) == e009['export_manifest_sha256'], 'Not the E009 native SVD export')
    report.update(manifest=file_record(args.manifest), settings=cfg, export_manifest=file_record(export),
        sources={p.name:file_record(p) for p in (Path(__file__), HERE/'h3_coarse16_attention.py',
            HERE/'h3_native_fp4_attention.py', HERE/'codebook_attention_sm120.py')},
        allocated_dit_calls=len(manifest['cases'])*cfg['num_inference_steps'],
        environment=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            attention_implementation=os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'],
            sdpa_enabled={name:getattr(torch.backends.cuda,name+'_sdp_enabled')()
                for name in ('flash','math','mem_efficient','cudnn')}))
    return manifest


def prepare_bindings(args, manifest, report, required):
    """Only actual input files/tensors and CPU packing; no model/source-tree audit."""
    pipe = old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['schedule_cpu'] = schedules(pipe, manifest['settings'])
    if not args.prepare_report.exists():
        require(not required, 'Preparation report not available')
        report['prepare_available'] = False
        precheck = args.report_dir/'prepare_check.json'
        lengths = []
        if precheck.exists():
            check = json.loads(precheck.read_text())
            for row in check.get('cases', []):
                contract = row.get('attention_contract')
                if contract:
                    lengths.append(contract['expected_cu'][1])
        return [], lengths
    prep = old.load_complete(args.prepare_report)
    if 'manifest' in prep:
        require(prep['manifest']['sha256'] == report['manifest']['sha256'], 'Prepare manifest changed')
    rows = {r['case_id']:r for r in prep['cases']}
    require(set(rows) == {r['case_id'] for r in manifest['cases']}, 'Prepared case set differs')
    bound, lengths = [], []
    for case in manifest['cases']:
        row = rows[case['case_id']]
        require((row['prompt_id'], row['seed']) == (case['prompt_id'], case['seed']), 'Prepared identity changed')
        record = file_record(row['file'])
        require(record['sha256'] == row['sha256'], 'Prepared input file SHA differs')
        cached = torch.load(row['file'], map_location='cpu', weights_only=True, mmap=True)
        require(cached['case_id'] == case['case_id'] and cached['seed'] == case['seed']
                and cached['prompt'] == case['prompt'], 'Prepared payload identity changed')
        embedding, tags = cached['embedding'], cached['text_token_tags']
        require(embedding.ndim == 2 and embedding.shape[1] == 5120 and embedding.dtype == torch.bfloat16,
                'Expected real BF16 text embedding [tokens,5120]')
        require(tags.shape == (embedding.shape[0],) and tags.unique().tolist() == [1],
                'Embedding bypass would change actual text tags')
        require(tensor_record(embedding)['sha256'] == row['embedding_sha256'], 'Embedding SHA differs')
        for key in ('video_latents','audio_latents'):
            require(cached[key].dtype == torch.bfloat16 and bool(torch.isfinite(cached[key]).all()), 'Invalid actual noise')
            require(tensor_record(cached[key])['sha256'] == row['initial_noise_sha256'][key], 'Actual shared noise SHA differs')
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(pipe, prompt_embeds=embedding,
            text_token_tags=tags, video_latents=cached['video_latents'], audio_latents=cached['audio_latents'])['packed']
        contract = dict(expected_cu=packed['cu_seqlens'].tolist(),
            expected_refiner_cu=[0,embedding.shape[0],embedding.shape[0]])
        if 'attention_contract' in row:
            require(contract == row['attention_contract'], 'Actual prepared layout differs')
        lengths.append(contract['expected_cu'][1])
        binding_devices = {key: str(cached[key].device) for key in
            ('embedding', 'text_token_tags', 'video_latents', 'audio_latents')}
        binding_devices['packed_cu_seqlens'] = str(packed['cu_seqlens'].device)
        bound.append(dict(case_id=case['case_id'], prepared=record, attention_contract=contract,
            initial_noise_sha256=row['initial_noise_sha256'], embedding_sha256=row['embedding_sha256'],
            binding_tensor_devices=binding_devices))
        del cached, packed, embedding, tags
    if args.phase == 'check':
        require(not torch.cuda.is_initialized(), 'CPU input binding initialized CUDA')
    report.update(prepare_available=True, prepared_reference=file_record(args.prepare_report), prepared_cases=bound)
    return bound, lengths


@torch.inference_mode()
def denoise(args, manifest, bound, report):
    budget(args, report)
    require(torch.cuda.get_device_capability() == (12,0), 'Native attention requires SM120')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(manifest['budget']['generation_gpus'][args.arm]),
            'Visible GPU differs from fixed arm allocation')
    torch.cuda.set_per_process_memory_fraction(min(1.,60*2**30/torch.cuda.get_device_properties(0).total_memory))
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    stage = args.data_dir/f'denoise_{args.arm}'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = old.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = old.make_h3_resident(pipe.dit)
    report['native_installation'] = old.install_native_h3(pipe.dit, Path(manifest['export_dir']),
        activation_packer=old.pack_activation_fast, chunk_rows=1024)
    require(report['native_installation']['target_count'] == 200
        and report['native_installation']['exact_roundtrip_count'] == 200, 'Incomplete native linear installation')
    router = install_h3_center_attention(pipe.dit, args.arm)
    report['attention_installation'] = router.manifest
    gc.collect()
    torch.cuda.empty_cache()
    original_step, original_forward = pipe.step, pipe.dit.forward
    cfg = manifest['settings']
    audit = old.RuntimeAudit()
    audit.phase = 'native'
    try:
        with audit.installed():
            for case, cpu in zip(manifest['cases'], bound, strict=True):
                budget(args, report)
                cached = torch.load(cpu['prepared']['file'], map_location='cpu', weights_only=True, mmap=True)
                directory = stage/case['case_id']
                directory.mkdir()
                info = dict(**case, variant=args.arm, status='running', prepared=cpu['prepared'],
                    initial_noise_sha256=cpu['initial_noise_sha256'], embedding_sha256=cpu['embedding_sha256'],
                    attention_contract=cpu['attention_contract'], steps=[], dit_calls=[])
                report['cases'].append(info)
                before_case = dict(audit.row())
                noise_calls, final = [], {}

                def cached_noise(_self, shape, seed=None, rand_device='cpu', rand_torch_dtype=torch.float32,
                                 device=None, torch_dtype=None):
                    index = len(noise_calls)
                    key = ('video_latents','audio_latents')[index] if index < 2 else None
                    require(key is not None and tuple(shape) == tuple(cached[key].shape) and seed == case['seed']
                        and rand_device == 'cpu' and rand_torch_dtype == torch.bfloat16, 'Noise initializer contract changed')
                    noise_calls.append(key)
                    return cached[key].clone().to(device=device or pipe.device, dtype=torch_dtype or pipe.torch_dtype)

                def resident_placement(_self, names):
                    if tuple(names) == tuple(pipe.in_iteration_models):
                        return
                    if list(names) == ['video_vae']:
                        require(len(info['steps']) == 40 and len(info['dit_calls']) == 20, 'Sampling completion count changed')
                        raise old.DenoiseComplete('Original sampler complete; decode in separate process')
                    raise RuntimeError(f'Unexpected resident component load: {names}')

                def observed_forward(_self, *pos, **kwargs):
                    budget(args, report, before_forward=True)
                    require(len(info['dit_calls']) < 20, 'Extra per-case DiT call')
                    before = dict(audit.row())
                    report['attempted_dit_calls'] += 1
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    with old.collect_fastpack_checks() as checks, router.forward_context(
                            **cpu['attention_contract'], diagnostics=False) as attention:
                        output = original_forward(*pos, **kwargs)
                        torch.cuda.synchronize()
                    elapsed = time.monotonic()-started
                    counts = {k:audit.row()[k]-before[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')}
                    low = args.arm != 'bf16'
                    require(counts == dict(sdpa_calls=52 if low else 102, scaled_mm_calls=200, disk_loads=0),
                            f'Unexpected native path: {counts}')
                    require(checks.summary['checked_calls'] == 200 and checks.summary['invalid_calls'] == 0,
                            'Invalid/incomplete fastpack checks')
                    summary = attention.summary
                    require(summary['fp4_calls'] == (50 if low else 0) and summary['finite_flag_count'] == 0,
                            'Unexpected attention route')
                    for x in output:
                        require(bool(torch.isfinite(x).all()), 'Nonfinite raw DiT output')
                    # Every layer is counted by the actual router; retain full layout once per case.
                    if not info['dit_calls']:
                        info['first_dit_attention_routes'] = summary['main_rows']
                    info['dit_calls'].append(dict(step=len(info['dit_calls']), counts=counts,
                        zero_sf_checks=checks.summary, attention={k:v for k,v in summary.items() if k!='main_rows'},
                        synchronized_dit_seconds=elapsed, finite=True))
                    report['complete_dit_calls'] += 1
                    return output

                def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
                    budget(args, report)
                    modality = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
                    require(modality is not None and 0 <= progress_id < 20, 'Unexpected scheduler call')
                    require(len(info['steps']) == 2*progress_id+MODALITIES.index(modality), 'Scheduler order changed')
                    if progress_id == 0:
                        require(tensor_record(latents)['sha256'] == cpu['initial_noise_sha256'][modality+'_latents'],
                                'Actual first sampler state differs from shared prepared noise')
                    updated = original_step(scheduler, latents, progress_id, noise_pred, **kwargs)
                    info['steps'].append(dict(step=progress_id, modality=modality,
                        timestep=float(scheduler.timesteps[progress_id]), sigma=float(scheduler.sigmas[progress_id]),
                        latent_before=tensor_scalars(latents), velocity=tensor_scalars(noise_pred),
                        latent_after=tensor_scalars(updated)))
                    if progress_id == 19:
                        final[modality+'_latents'] = updated.detach().cpu()
                    if modality == 'audio':
                        save(report, args.output)
                        if progress_id % 5 == 4:
                            print(f'E038 {args.arm} {case["case_id"]}: {progress_id+1}/20 steps', flush=True)
                    return updated

                pipe.generate_noise = types.MethodType(cached_noise, pipe)
                pipe.load_models_to_device = types.MethodType(resident_placement, pipe)
                pipe.step = types.MethodType(observed_step, pipe)
                pipe.dit.forward = types.MethodType(observed_forward, pipe.dit)
                started = time.monotonic()
                print(f'E038 {args.arm} {case["case_id"]}: original free rollout', flush=True)
                try:
                    pipe(prompt=None, text_embedding=cached['embedding'], seed=case['seed'],
                        height=cfg['height'], width=cfg['width'], num_frames=cfg['num_frames'],
                        num_inference_steps=cfg['num_inference_steps'], cfg_scale=cfg['cfg_scale'],
                        flow_shift=cfg['flow_shift'], audio_flow_shift=cfg['audio_flow_shift'],
                        rand_device=cfg['rand_device'], tiled=cfg['tiled'],
                        tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
                except old.DenoiseComplete:
                    pass
                else:
                    raise RuntimeError('Expected stop at original VAE boundary')
                require(noise_calls == ['video_latents','audio_latents']
                    and set(final) == {'video_latents','audio_latents'}, 'Incomplete shared-input/final output')
                require(actual_schedules(pipe) == report['schedule_cpu'], 'Runtime schedules changed')
                path = directory/'final_latents.pt'
                torch.save(final, path)
                info.update(status='complete', final_latents=file_record(path), final_tensors=tree_signature(final),
                    schedule=actual_schedules(pipe), seconds_including_diagnostics=time.monotonic()-started,
                    runtime_audit={k:audit.row()[k]-before_case[k] for k in ('sdpa_calls','scaled_mm_calls','disk_loads')})
                save(report, args.output)
                del cached, final
                gc.collect()
                torch.cuda.empty_cache()
        require(report['complete_dit_calls'] == report['attempted_dit_calls'] == report['allocated_dit_calls'],
                'Incomplete allocated rollout')
        report['runtime_audit'] = dict(audit.row())
        report['actual_fp4_attention_calls'] = sum(d['attention']['fp4_calls'] for c in report['cases'] for d in c['dit_calls'])
        report['status'] = 'complete'
    finally:
        router.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check','denoise'), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--prepare-report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.report_writable = False
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(experiment='E038', phase=args.phase, arm=args.arm, status='running', cases=[],
        attempted_dit_calls=0, complete_dit_calls=0, deadline_unix=args.deadline_unix,
        scope='Same native SVD linear weights; only attention center choice changes. No full-model SP speed claim.')
    started = time.monotonic()
    try:
        manifest = read_manifest(args, report)
        bound, lengths = prepare_bindings(args, manifest, report, required=args.phase=='denoise')
        if args.phase == 'check':
            report['geometry_check'] = cpu_geometry_check(lengths)
            report.update(status='complete', cuda_initialized=torch.cuda.is_initialized())
            require(not report['cuda_initialized'], 'CPU check initialized CUDA')
        else:
            denoise(args, manifest, bound, report)
            budget(args, report)
    except BaseException:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        if args.report_writable:
            save(report, args.output)
        print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
