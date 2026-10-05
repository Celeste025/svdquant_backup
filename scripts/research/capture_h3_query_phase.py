#!/usr/bin/env python3
"""E018 single-forward QKV capture from the frozen E017 p36 block-mean s14 state.

The CPU check executes only original input packing/model_fn up to a stub DiT.
The GPU phase runs exactly one original full DiT; this file has no kernel probe.
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
import run_h3_fp4_attention_video as prior
import probe_h3_plain_baseline as base
import torch
from minimax_h3_svdquant_common import tree_device
from h3_native_fp4_attention import install_h3_fp4_attention

DATA = Path('/data1/models/svdquant-wjq/research/20261002/E018')
REPORTS = ROOT/'results/research/E018'
E017 = ROOT/'results/research/E017'
PLAN = ROOT/'research_state/06_experiments/E018_h3_query_phase_plan.md'
BLOCKS = (0, 24, 48)
MODALITIES = ('video', 'audio')
file_record, tensor_record, tree_signature = base.file_record, base.tensor_record, base.tree_signature
save, sha256, require = prior.save, prior.sha256, prior.require


def deadline(args, report, before_forward=False):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'Capture requires the unexpired original shared absolute deadline')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '5', 'Only physical GPU5 is authorized')
    if before_forward:
        require(report['attempted_dit_calls'] == 0, 'Exactly one attempted DiT call is permitted')
    prior.old.memory_guard()


def inputs_and_binding(args, report):
    inherited_args = types.SimpleNamespace(phase='check', manifest=prior.MANIFEST, plan=prior.PLAN,
        data_dir=prior.DATA, report_dir=prior.REPORTS)
    inherited = {}
    manifest, prep, _ = prior.prerequisites(inherited_args, inherited)
    checked = prior.old.load_complete(E017/'E017_check_block_mean.json')
    require(checked['cuda_initialized'] is False and checked['sources'] == inherited['sources'],
            'Inherited E017 CPU/source binding changed')
    frozen = prior.old.load_complete(E017/'frozen_contract.json')
    for path, digest in frozen['files'].items():
        require(sha256(path) == digest, f'E017 frozen source drift: {path}')
    report_path = E017/'E017_denoise_block_mean.json'
    reference = prior.old.load_complete(report_path)
    require(reference['arm'] == 'block_mean' and reference['complete_dit_calls'] == 40 and
            reference['sources'] == checked['sources'], 'Incomplete/mismatched E017 rollout')
    case = next(c for c in reference['cases'] if c['prompt_id'] == 36)
    require(case['status'] == 'complete' and case['seed'] == 59526, 'Wrong source case')
    original = next(c for c in prep['cases'] if c['prompt_id'] == 36)
    embedding = torch.load(base.verify_file(case['prepared']), map_location='cpu', weights_only=True, mmap=True)
    require(case['prepared']['sha256'] == original['sha256'], 'Prepared embedding/noise source changed')
    state, references = {}, {}
    for modality in MODALITIES:
        row = next(r for r in case['steps'] if r['step'] == 14 and r['modality'] == modality)
        source = {key: row[key] for key in ('file', 'sha256', 'bytes')}
        payload = torch.load(base.verify_file(source), map_location='cpu', weights_only=True, mmap=True)
        require(tree_signature(payload) == row['tensors'] and row['finite'] is True, 'Source step tensor SHA changed')
        state[modality] = {k: payload[k] for k in ('latents_before', 'timestep', 'sigma')}
        references[modality] = {'file': source, 'tensors': row['tensors']}
    cpu = next(c for c in checked['cases'] if c['prompt_id'] == 36)
    contract = cpu['attention_contract']
    require(contract == {'expected_cu': [0, 22539, 22592], 'expected_refiner_cu': [0, 813, 813]},
            'Source attention geometry changed')
    call = case['dit_calls'][14]
    require(call['step'] == 14 and call['counts'] == dict(sdpa_calls=52, scaled_mm_calls=200, disk_loads=0)
            and call['attention']['fp4_calls'] == 50 and call['finite'] is True, 'Source call contract changed')
    sources = dict(inherited['sources'])
    sources.update({str(p.resolve()): file_record(p) for p in (Path(__file__), PLAN)})
    report.update(sources=sources, environment=inherited['environment'],
        plan=file_record(PLAN),
        inherited_asset_reference=inherited['prepared_reference'], inherited_assets=inherited['inherited_assets'],
        source_report=file_record(report_path), source_check=file_record(E017/'E017_check_block_mean.json'),
        source_freeze=file_record(E017/'frozen_contract.json'), source_manifest=file_record(prior.MANIFEST),
        source_steps=references, prepared=case['prepared'], export_manifest=inherited['export_manifest'],
        case=dict(prompt_id=36, seed=59526, step=14, arm='block_mean'), attention_contract=contract,
        expected_dit_inputs=call['actual_dit_inputs'], expected_raw_outputs=call['raw_outputs'],
        expected_velocities={m: references[m]['tensors']['noise_pred'] for m in MODALITIES})
    value = dict(embedding=embedding['embedding'], text_token_tags=embedding['text_token_tags'], state=state)
    return manifest, value


def construct_call(value):
    pipe = prior.old.MiniMaxH3Pipeline(device='cpu', torch_dtype=torch.bfloat16)
    packed = base.make_packed(pipe, value['embedding'], value['text_token_tags'], value['state'])
    captured = []
    class ReachedDiT(Exception):
        pass
    def stub(*pos, **kwargs):
        captured.append({'args': pos, 'kwargs': kwargs})
        raise ReachedDiT()
    try:
        base.model_fn_minimax_h3(dit=stub, video_latents=value['state']['video']['latents_before'],
            audio_latents=value['state']['audio']['latents_before'], packed=packed,
            prompt_embeds=value['embedding'], timestep_video=value['state']['video']['timestep'].reshape(1),
            timestep_audio=value['state']['audio']['timestep'].reshape(1))
    except ReachedDiT:
        pass
    require(len(captured) == 1, 'Expected precisely one CPU input-construction stub invocation')
    return packed, captured[0]


def cpu_structure(value, report):
    packed, raw_call = construct_call(value)
    actual = tree_signature(raw_call)
    require(actual == report['expected_dit_inputs'], 'CPU-reconstructed actual DiT inputs differ from E017 bytes')
    positions = {name: packed[key] for name, key in (('video', 'img_pos'), ('audio', 'audio_pos'), ('text', 'text_pos'))}
    require(torch.equal(positions['video'], torch.arange(1227, 22539)) and
            torch.equal(positions['audio'], torch.arange(813, 1227)) and
            torch.equal(positions['text'], torch.arange(813)), 'Unexpected multimodal layout/conditions')
    report.update(input_signature=tree_signature(value), actual_dit_inputs=actual,
        input_reconstruction_exact=True, modality_positions={k: tensor_record(v) for k, v in positions.items()},
        position_ranges={k: dict(first=int(v[0]), last=int(v[-1]), length=v.numel()) for k, v in positions.items()},
        capture_shapes={str(b): [22539, 56, 128] for b in BLOCKS},
        estimated_tensor_bytes=3*4*22539*56*128*2,
        capture_scope='Original post-QK-norm/RoPE helper Q/K/V and complete router output before out_proj; valid segment only')
    return packed, raw_call, positions


@torch.inference_mode()
def capture(args, manifest, value, report):
    check_path = args.report_dir/'capture_check.json'
    checked = prior.old.load_complete(check_path)
    require(checked['sources'] == report['sources'] and checked['environment'] == report['environment'] and
            checked['cuda_initialized'] is False and checked['input_signature'] == report['input_signature'] and
            checked['actual_dit_inputs'] == report['actual_dit_inputs'], 'Capture source/input differs from CPU check')
    report['cpu_check'] = file_record(check_path)
    deadline(args, report)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    require(torch.cuda.get_device_capability() == (12, 0), 'Native capture requires SM120')
    report['device'] = dict(name=torch.cuda.get_device_name(), capability=[12, 0], visible_devices='5')
    stage = args.data_dir/'capture'
    stage.mkdir(parents=True, exist_ok=False)
    pipe = prior.old.load_h3_pipeline(full=False, vram_limit_gib=30.)
    pipe.load_models_to_device(['dit'])
    pipe.dit.eval()
    report['resident_conversion'] = prior.old.make_h3_resident(pipe.dit)
    identities = base.non_target_identity(pipe.dit)
    report['native_installation'] = prior.old.install_native_h3(pipe.dit, Path(manifest['export_dir']),
        activation_packer=prior.old.pack_activation_fast, chunk_rows=1024)
    require(report['native_installation']['target_count'] == report['native_installation']['exact_roundtrip_count'] == 200,
            'Incomplete native installation')
    require(base.non_target_identity(pipe.dit) == identities, 'Changed non-target tensors')
    from diffsynth.core.vram.layers import AutoTorchModule
    require(not any(isinstance(m, AutoTorchModule) for m in pipe.dit.modules()), 'Residual offload module')
    router = install_h3_fp4_attention(pipe.dit, mode='block_mean')
    report['attention_installation'] = router.manifest
    gc.collect()
    torch.cuda.empty_cache()
    packed, raw_call, positions = cpu_structure(value, report)
    gpu_call = tree_device(raw_call, 'cuda')
    require(tree_signature(gpu_call) == report['actual_dit_inputs'], 'Input device transfer changed bytes')
    original_dispatch = router.comfy._sdpa_varlen_attention
    report['cases'] = []
    n = report['attention_contract']['expected_cu'][1]

    def observe_helper(q, k, v, cu_seqlens, softmax_scale):
        block = router._active_main.get()
        if block not in BLOCKS:
            return original_dispatch(q, k, v, cu_seqlens, softmax_scale)
        require(block not in {r['block'] for r in report['cases']}, 'Duplicate capture block')
        payload = {name: tensor[:n].detach().cpu().contiguous() for name, tensor in zip(('q', 'k', 'v'), (q, k, v))}
        output = original_dispatch(q, k, v, cu_seqlens, softmax_scale)
        payload['router_output'] = output[:n].detach().cpu().contiguous()
        for tensor in payload.values():
            require(tuple(tensor.shape) == (n, 56, 128) and tensor.dtype == torch.bfloat16 and
                    bool(torch.isfinite(tensor).all()), 'Invalid captured tensor')
        payload.update(block=block, positions=positions, cu_seqlens=cu_seqlens.detach().cpu(),
            valid_length=n, video_start=1227, video_tokens=21312, scale=softmax_scale, source_case=report['case'],
            source_report=report['source_report'], source_steps=report['source_steps'])
        path = stage/f'block{block}.pt'
        torch.save(payload, path)
        report['cases'].append(dict(block=block, artifact=file_record(path),
            tensors=tree_signature(payload), finite=True))
        save(report, args.output)
        print(f'E018 captured block{block}: {path}', flush=True)
        return output

    audit = prior.old.RuntimeAudit()
    audit.phase = 'native'
    try:
        router.comfy._sdpa_varlen_attention = observe_helper
        deadline(args, report, before_forward=True)
        with audit.installed(), prior.old.collect_fastpack_checks() as checks, router.forward_context(
                **report['attention_contract'], diagnostics=False) as attn:
            report['attempted_dit_calls'] += 1
            outputs = pipe.dit(*gpu_call['args'], **gpu_call['kwargs'])
            torch.cuda.synchronize()
        report['complete_dit_calls'] = 1
    finally:
        router.comfy._sdpa_varlen_attention = original_dispatch
        router.close()
    runtime = audit.row()
    require({key: runtime[key] for key in ('sdpa_calls', 'scaled_mm_calls', 'disk_loads')}
            == dict(sdpa_calls=52, scaled_mm_calls=200, disk_loads=0), 'Actual runtime counts differ')
    require(checks.summary['checked_calls'] == 200 and checks.summary['invalid_calls'] == 0 and
            attn.summary['fp4_calls'] == 50, 'Native packing/attention contract failed')
    raw, raw_records = base.cpu_outputs(outputs)
    require(raw_records == report['expected_raw_outputs'], 'Full original DiT raw output differs from E017 SHA')
    from diffsynth.pipelines.minimax_h3_audio_video import unpatchify_video, unpack_audio
    f, h, w = value['state']['video']['latents_before'].shape[2:]
    audio = value['state']['audio']['latents_before']
    velocities = dict(video=-unpatchify_video(raw['video'], f, h, w),
                      audio=-unpack_audio(raw['audio'], audio.shape[0], audio.shape[-1]))
    velocity_records = tree_signature(velocities)
    require(velocity_records == report['expected_velocities'], 'Original unpack/sign velocity differs from E017 SHA')
    require([r['block'] for r in report['cases']] == list(BLOCKS), 'Capture block set/order differs')
    replay_path = stage/'full_replay.pt'
    torch.save(dict(raw_outputs=raw, velocities=velocities, actual_dit_inputs=report['actual_dit_inputs']), replay_path)
    report.update(status='complete', raw_outputs=raw_records, velocities=velocity_records,
        raw_replay_exact=True, velocity_replay_exact=True, full_replay=file_record(replay_path),
        runtime_audit=runtime, zero_sf_checks=checks.summary, attention=attn.summary,
        arithmetic_scope='Capture-only wrapper delegates original helper unchanged; no query permutation applied')
    deadline(args, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('check', 'capture', 'run'))
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    args.output = args.output or args.report_dir/('capture_check.json' if args.phase == 'check' else 'capture_run.json')
    require(not args.output.exists(), f'Refusing to overwrite prior report: {args.output}')
    report = dict(experiment='E018', phase=args.phase, status='running', attempted_dit_calls=0,
        complete_dit_calls=0, deadline_unix=args.deadline_unix,
        scope='One input-chain replay and three fixed QKV captures; no mechanism or quality conclusion')
    started = time.monotonic()
    try:
        manifest, value = inputs_and_binding(args, report)
        cpu_structure(value, report)
        if args.phase == 'check':
            require(not torch.cuda.is_initialized(), 'CPU structure check initialized CUDA')
            report.update(status='complete', cuda_initialized=False)
        else:
            capture(args, manifest, value, report)
    except Exception:
        report.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save(report, args.output)
        print(json.dumps(dict(status=report['status'], report=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
