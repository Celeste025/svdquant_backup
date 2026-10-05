#!/usr/bin/env python3
"""E038 shared text/noise preparation and unchanged H3 VAE decoding.

Only this orchestration is new: E010 supplies the text encoder, CPU noise,
and pipeline/VAE implementations. No DiT or attention kernel is called here.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import run_h3_native_paired_video as old
import probe_h3_plain_baseline as base
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'research_state/06_experiments/E038_center_video_manifest.json'
RD = ROOT/'results/research/E038'
DATA = Path('/data1/models/svdquant-wjq/research/20261003/E038')
REFERENCE = ROOT/'results/research/E010_h3_prepare.json'
ARMS = ('bf16', 'global_mean', 'coarse16')
file_record, tensor_record, tree_signature = base.file_record, base.tensor_record, base.tree_signature


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_prepared(row):
    return torch.load(base.verify_file(row), map_location='cpu', weights_only=True, mmap=True)


def schedule(pipe, settings):
    pipe.scheduler.set_timesteps(settings['num_inference_steps'], shift=settings['flow_shift'])
    pipe.scheduler_audio.set_timesteps(settings['num_inference_steps'], shift=settings['audio_flow_shift'])
    return {m: dict(timesteps=s.timesteps.tolist(), sigmas=s.sigmas.tolist())
            for m, s in (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}


def packed_contract(pipe, embedding, tags, noise):
    packed = MiniMaxH3Unit_PackedSequenceBuilder().process(pipe, prompt_embeds=embedding,
        text_token_tags=tags, video_latents=noise['video_latents'], audio_latents=noise['audio_latents'])['packed']
    return dict(expected_cu=packed['cu_seqlens'].tolist(),
                expected_refiner_cu=[0, embedding.shape[0], embedding.shape[0]])


def prerequisites(args, report):
    manifest = json.loads(args.manifest.read_text())
    require(manifest['experiment'] == 'E038' and manifest['arms'] == list(ARMS), 'Wrong E038 protocol')
    require(args.data_dir == Path(manifest['data_dir']) and args.report_dir == Path(manifest['report_dir']),
            'Data/report directories differ from manifest')
    require(sys.executable == manifest['python_decode'], 'Prepare/decode must use the recovered H3 environment')
    cases = manifest['cases']
    require(len(cases) == 8 and len({c['case_id'] for c in cases}) == 8, 'Expected eight unique cases')
    require(len({c['prompt_id'] for c in cases}) == 4, 'Expected four prompts')
    prompt_source = json.loads(Path(manifest['prompt_source']['file']).read_text())
    require(old.sha256(manifest['prompt_source']['file']) == manifest['prompt_source']['sha256'], 'Prompt source changed')
    for case in cases:
        require(case['prompt_id'] == case['source_index_zero_based'], 'VBench zero-based ID changed')
        require(prompt_source[case['prompt_id']]['prompt_en'] == case['prompt'], 'Selected source text differs')
        require(hashlib.sha256(case['prompt'].encode()).hexdigest() == case['prompt_sha256'], 'Prompt digest differs')
        require(case['replica'] in (0, 1), 'Invalid replica')
    reference = old.load_complete(REFERENCE)
    require(manifest['settings'] == reference['settings'], 'Original E010 sampling/decoder settings changed')
    paths = {Path(__file__), args.manifest, ROOT/'research_state/06_experiments/E038_center_video_plan.md', Path(old.__file__), Path(base.__file__),
        ROOT/'scripts/minimax_h3_svdquant_common.py',
        old.DS/'diffsynth/pipelines/minimax_h3_audio_video.py',
        old.DS/'diffsynth/models/minimax_h3_text_encoder.py',
        old.DS/'diffsynth/models/minimax_h3_video_vae.py',
        old.DS/'diffsynth/models/minimax_h3_audio_vae.py',
        old.DS/'diffsynth/core/vram/layers.py', old.DS/'diffsynth/utils/data/audio_video.py'}
    sources = {str(p.resolve()): file_record(p) for p in sorted(paths)}
    report.update(manifest=file_record(args.manifest), sources=sources, settings=manifest['settings'],
        asset_policy='Existing verified H3 assets and original E010 loader configuration; no repeated asset scan.',
        export_manifest=file_record(Path(manifest['export_dir'])/'manifest.json'),
        environment=dict(python=sys.executable, torch=torch.__version__, torch_cuda=torch.version.cuda,
                         attention=os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION']))
    require(torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8', 'Recovered environment changed')
    if args.phase == 'check':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU check requires CUDA hidden')
    else:
        gpu = manifest['budget']['generation_gpus']['bf16' if args.phase == 'prepare' else args.arm]
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu), 'Wrong stage GPU')
        checked = old.load_complete(args.check_report)
        require(checked['cuda_initialized'] is False and checked['manifest'] == report['manifest']
                and checked['sources'] == sources, 'CPU source/manifest check changed')
        report['cpu_check_reference'] = file_record(args.check_report)
    return manifest


def cpu_check(manifest, report):
    from transformers import AutoProcessor
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va
    processor = AutoProcessor.from_pretrained(str(old.MODEL_ROOT/'FL2VA/processor'), local_files_only=True)
    pipe = old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    report['schedule_cpu'] = schedule(pipe, manifest['settings'])
    tokenized = {}
    report['cases'] = []
    for case in manifest['cases']:
        pid = case['prompt_id']
        if pid not in tokenized:
            ids, tags = presentation_t2va(processor.tokenizer, case['prompt'])
            require(tags.unique().tolist() == [1], 'Cached embedding bypass requires positive text tags')
            tokenized[pid] = (ids.numel(), tags.reshape(-1).cpu())
        tokens, tags = tokenized[pid]
        noise = old.initial_noise(manifest['settings'], case['seed'])
        require(all(bool(t.isfinite().all()) for t in noise.values()), 'Invalid CPU noise')
        # Only geometry is inferred here; no claim that these dummy embeddings
        # are real encoder outputs. Prepare checks the actual BF16 shape/tags.
        dummy = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
        contract = packed_contract(pipe, dummy, tags, noise)
        require(contract['expected_cu'][1] < contract['expected_cu'][2], 'Inherited attention router requires nonempty model padding')
        report['cases'].append(dict(case_id=case['case_id'], prompt_id=pid, seed=case['seed'],
            text_tokens=tokens, text_token_tags=tensor_record(tags),
            initial_noise={k: tensor_record(v) for k, v in noise.items()}, attention_contract=contract))
    report['scalar_fixture'] = {m: tensor_record(s.timesteps[0]) for m, s in
        (('video', pipe.scheduler), ('audio', pipe.scheduler_audio))}
    require(not torch.cuda.is_initialized(), 'CPU preflight initialized CUDA')
    report.update(status='complete', cuda_initialized=False, unique_text_encodings_planned=len(tokenized),
        fixture_scope='Actual tokenizer/noise and packed geometry; real encoder embedding checked during prepare.')


def budget(args):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix, 'Missing/expired GPU stage deadline')
    old.memory_guard()


@torch.inference_mode()
def prepare(args, manifest, report):
    budget(args)
    stage = args.data_dir/'prepare'
    stage.mkdir(parents=True, exist_ok=False)
    checked = old.load_complete(args.check_report)
    rows = {c['case_id']: c for c in checked['cases']}
    shards = sorted((old.MODEL_ROOT/'FL2VA/text_encoder').glob('model*.safetensors'))
    pipe = old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[old.ModelConfig(path=[str(p) for p in shards], **old.disk_config())],
        processor_config=old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/processor')), vram_limit=30.)
    require(pipe.dit is None and pipe.video_vae is None and pipe.audio_vae is None, 'Prepare must contain only TE')
    pipe.text_encoder.eval()
    unit = old.MiniMaxH3Unit_PromptEmbedder()
    cpu_pipe = old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    embeddings = {}
    report['cases'] = []
    report['text_encodings'] = []
    for case in manifest['cases']:
        budget(args)
        started = time.monotonic()
        pid = case['prompt_id']
        cpu = rows[case['case_id']]
        if pid not in embeddings:
            print(f'E038 text encode VBench[{pid}]', flush=True)
            embedded = unit.process(pipe, prompt=case['prompt'],
                height=manifest['settings']['height'], width=manifest['settings']['width'])
            embedding = embedded['prompt_embeds'].detach().cpu()
            tags = embedded['text_token_tags'].detach().cpu()
            require(embedding.shape == (cpu['text_tokens'], 5120) and embedding.dtype == torch.bfloat16,
                    'Actual text embedding rank/axes changed')
            require(bool(embedding.isfinite().all()) and tensor_record(tags) == cpu['text_token_tags'],
                    'Actual embedding/tags invalid')
            embeddings[pid] = (embedding, tags)
            report['actual_text_encoder_calls'] += 1
            report['text_encodings'].append(dict(prompt_id=pid, prompt_sha256=case['prompt_sha256'],
                embedding=tensor_record(embedding), text_token_tags=tensor_record(tags),
                seconds=time.monotonic()-started))
        embedding, tags = embeddings[pid]
        noise = old.initial_noise(manifest['settings'], case['seed'])
        require({k: tensor_record(v) for k, v in noise.items()} == cpu['initial_noise'], 'CPU noise changed')
        contract = packed_contract(cpu_pipe, embedding, tags, noise)
        require(contract == cpu['attention_contract'], 'Actual encoder packed layout differs from preflight')
        payload = {**case, 'embedding': embedding, 'text_token_tags': tags, **noise}
        path = stage/f'{case["case_id"]}.pt'
        torch.save(payload, path)
        report['cases'].append(dict(**case, **file_record(path), embedding_shape=list(embedding.shape),
            embedding_sha256=tensor_record(embedding)['sha256'], text_token_tags_sha256=tensor_record(tags)['sha256'],
            initial_noise_sha256={k: tensor_record(v)['sha256'] for k, v in noise.items()},
            attention_contract=contract, seconds=time.monotonic()-started))
        old.save(report, args.output)
        del payload, noise
    require(report['actual_text_encoder_calls'] == 4 and len(report['cases']) == 8, 'Incomplete shared preparation')
    report.update(status='complete', schedule_cpu=checked['schedule_cpu'])


@torch.inference_mode()
def decode(args, manifest, report):
    import av
    from diffsynth.utils.data.audio_video import write_video_audio
    budget(args)
    source = args.report_dir/f'denoise_{args.arm}.json'
    denoiser = old.load_complete(source)
    require(denoiser['manifest'] == report['manifest'] and denoiser['arm'] == args.arm, 'Denoiser binding differs')
    require(denoiser['complete_dit_calls'] == 160 and len(denoiser['cases']) == 8, 'Incomplete denoising allocation')
    prep = old.load_complete(args.prepare_report)
    require(prep['manifest'] == report['manifest'], 'Prepared manifest differs')
    report.update(denoiser_reference=file_record(source), prepared_reference=file_record(args.prepare_report), av_version=av.__version__)
    stage = args.data_dir/f'decode_{args.arm}'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'), **old.disk_config()),
                       old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/audio_vae/model.safetensors'), **old.disk_config())],
        processor_config=None, vram_limit=30.)
    require(pipe.dit is None and pipe.text_encoder is None and type(pipe.video_vae).__name__ == 'MiniMaxH3VideoVAE'
            and pipe.audio_vae.sample_rate == 32000, 'Decode must contain original VAEs only')
    pipe.video_vae.eval(); pipe.audio_vae.eval()
    cfg = manifest['settings']
    report['cases'] = []
    for case, row in zip(manifest['cases'], denoiser['cases'], strict=True):
        budget(args); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        require(row['status'] == 'complete' and all(row[k] == case[k] for k in ('case_id','prompt_id','seed','prompt')),
                'Denoiser case identity changed')
        latents = load_prepared(row['final_latents'])
        require(tree_signature(latents) == row['final_tensors'], 'Final latent tensor signatures differ')
        pipe.load_models_to_device(['video_vae'])
        recon = pipe.video_vae.decode_video(latents['video_latents'].cuda(), dtype=torch.bfloat16,
            tiled=cfg['tiled'], tile_size=cfg['tile_size'], tile_overlap=cfg['tile_overlap'])
        require(bool(recon.isfinite().all()), 'Nonfinite decoded video')
        video = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
        del recon
        require(len(video) == cfg['num_frames'] and all(im.size == (cfg['width'], cfg['height']) for im in video), 'Video shape differs')
        budget(args); pipe.load_models_to_device(['audio_vae'])
        waveform = pipe.audio_vae.decode_audio(latents['audio_latents'].cuda(), dtype=torch.bfloat16)
        audio = pipe.output_audio_format_check(waveform)
        require(bool(audio.isfinite().all()) and audio.ndim == 2 and audio.shape[0] == 2, 'Invalid stereo PCM')
        path = stage/f'{case["case_id"]}.mp4'
        write_video_audio(video, audio, str(path), fps=cfg['fps'], audio_sample_rate=cfg['audio_sample_rate'])
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
            vs, aus = container.streams.video[0], container.streams.audio[0]
            media = dict(frames=frames, width=vs.width, height=vs.height, fps=float(vs.average_rate),
                audio_streams=len(container.streams.audio), audio_sample_rate=aus.codec_context.sample_rate,
                audio_channels=aus.codec_context.channels)
        require(media == dict(frames=124, width=1024, height=576, fps=24., audio_streams=1,
                             audio_sample_rate=32000, audio_channels=2), 'Encoded media contract differs')
        pcm_path = path.with_suffix('.audio.pt')
        torch.save({'waveform': audio.detach().cpu(), 'sample_rate': 32000}, pcm_path)
        result = dict(**case, trajectory_id=case['case_id']+'_'+args.arm, status='complete', variant=args.arm,
            settings=cfg, video=file_record(path), video_path=str(path), video_sha256=old.sha256(path), media=media,
            final_latents=row['final_latents'], final_tensors=row['final_tensors'], audio_pcm_file=file_record(pcm_path),
            audio_pcm=tensor_record(audio), audio_samples_before_aac=audio.shape[-1],
            seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        old.save(result, path.with_suffix('.json')); report['cases'].append(result)
        report['actual_video_vae_calls'] += 1; report['actual_audio_vae_calls'] += 1
        old.save(report, args.output)
        del video, audio, waveform, latents
        gc.collect(); torch.cuda.empty_cache()
    report['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check','prepare','decode'), required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--prepare-report', type=Path)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    require(args.phase != 'decode' or args.arm is not None, 'Decode requires --arm')
    args.prepare_report = args.prepare_report or args.report_dir/'prepare.json'
    args.check_report = args.check_report or args.report_dir/'prepare_check.json'
    name = 'prepare_check' if args.phase == 'check' else ('prepare' if args.phase == 'prepare' else 'decode_'+args.arm)
    args.output = args.output or args.report_dir/(name+'.json')
    require(not args.output.exists(), 'Refusing to overwrite an earlier result/failed attempt')
    report = dict(experiment='E038', phase=args.phase, arm=args.arm, status='running',
        actual_dit_calls=0, actual_text_encoder_calls=0, actual_video_vae_calls=0, actual_audio_vae_calls=0,
        deadline_unix=args.deadline_unix)
    started = time.monotonic()
    try:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
        manifest = prerequisites(args, report)
        if args.phase == 'check': cpu_check(manifest, report)
        elif args.phase == 'prepare': prepare(args, manifest, report)
        else: decode(args, manifest, report)
        if args.phase != 'check': budget(args)
    except BaseException:
        report.update(status='failed_preserved', error=traceback.format_exc()); raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = max([torch.cuda.max_memory_allocated()] +
                [r.get('peak_allocated_bytes', 0) for r in report.get('cases', [])])
            report['peak_reserved_bytes'] = max([torch.cuda.max_memory_reserved()] +
                [r.get('peak_reserved_bytes', 0) for r in report.get('cases', [])])
        old.save(report, args.output)
        print(json.dumps(dict(status=report['status'], output=str(args.output))), flush=True)


if __name__ == '__main__': main()
