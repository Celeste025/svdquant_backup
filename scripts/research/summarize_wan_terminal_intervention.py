#!/usr/bin/env python3
"""CPU FP64 readout of E045 actual terminal intervention and four decoder corners.

Raw RGB stays memory-mapped; arithmetic is bounded to one latent slice/frame.
No quality score, cross-space norm ratio, or acceptance threshold is defined.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/E045'
PLAN = ROOT / 'research_state/06_experiments/E045_wan_terminal_intervention_manifest.json'
CORNERS = ('teacher', 'native_terminal', 'first_only', 'rest_only')
EFFECTS = ('first_only', 'rest_only', 'full', 'interaction')
EXPECTED = dict(teacher_dit=100, native_dit=2, native_gemm=600, dit_sdpa=6120,
                fastpack_checks=600, teacher_scheduler=50, shadow_scheduler=1,
                candidate_scheduler=1, public_vae_decode=4, decoder_chunks=84, text_encoder=0)


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return dict(file=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def tensor_signature(x, torch, clamp=False):
    """Hash contiguous storage order without constructing a second RGB tensor."""
    require(x.device.type == 'cpu', 'Expected CPU artifact')
    x = x.contiguous()
    digest, finite = hashlib.sha256(), True
    flat = x.reshape(-1)
    for begin in range(0, x.numel(), 1 << 20):
        part = flat[begin:begin+(1 << 20)]
        if clamp:
            part = part.clamp(-1, 1)
        finite = finite and bool(part.isfinite().all())
        digest.update(memoryview(part.view(torch.uint8).numpy()).cast('B'))
    return dict(shape=list(x.shape), dtype=str(x.dtype), finite=finite, sha256=digest.hexdigest())


def curve_summary(values, elements_per_index):
    return dict(mse_by_index=values, first_index_mse=values[0],
                rest_indices_mean_mse=sum(values[1:]) / (len(values)-1),
                all_indices_mean_mse=sum(values) / len(values),
                elements_per_index=elements_per_index, indices=len(values))


def pair_curve(x, y, torch):
    require(x.shape == y.shape and x.ndim == 5, 'Matching B,C,T,H,W pair required')
    values, maximum, changed = [], 0., 0
    for index in range(x.shape[2]):
        a, b = x[:, :, index].double(), y[:, :, index].double()
        delta = a-b
        values.append(float(delta.square().mean()))
        maximum = max(maximum, float(delta.abs().max()))
        changed += int((a != b).sum())
    result = curve_summary(values, x[:, :, 0].numel())
    result.update(max_abs=maximum, changed_elements=changed, equal=changed == 0)
    return result


def effect_curves(corners, torch, clamp=False):
    shape = corners['teacher'].shape
    require(all(x.shape == shape for x in corners.values()) and len(shape) == 5, 'Corner shapes')
    values = {name: [] for name in EFFECTS}
    for index in range(shape[2]):
        slices = {name: x[:, :, index].double() for name, x in corners.items()}
        if clamp:
            slices = {name: x.clamp(-1, 1) for name, x in slices.items()}
        b, q, first, rest = (slices[name] for name in CORNERS)
        differences = dict(first_only=first-b, rest_only=rest-b, full=q-b,
                           interaction=(q-first)-(rest-b))
        for name, delta in differences.items():
            values[name].append(float(delta.square().mean()))
    return {name: curve_summary(curve, corners['teacher'][:, :, 0].numel())
            for name, curve in values.items()}


def check_corner_identity(c, torch):
    b, q, first, rest = (c[name] for name in CORNERS)
    checks = dict(first_t0_from_native=torch.equal(first[:, :, :1], q[:, :, :1]),
                  first_rest_from_teacher=torch.equal(first[:, :, 1:], b[:, :, 1:]),
                  rest_t0_from_teacher=torch.equal(rest[:, :, :1], b[:, :, :1]),
                  rest_rest_from_native=torch.equal(rest[:, :, 1:], q[:, :, 1:]))
    require(all(checks.values()), 'Saved corners violate exact replacement identities')
    return checks


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_cpu_libraries():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU readout requires CUDA_VISIBLE_DEVICES empty')
    import torch
    torch.set_num_threads(4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    return torch, plt


def check(args, torch):
    plan = json.loads(args.manifest.read_text())
    require(plan['experiment'] == 'E045' and len(plan['cases']) == 2
            and tuple(plan['corners']) == CORNERS, 'Fixed two-case four-corner plan')
    require(plan['readout']['latent_slices'] == 21 and plan['readout']['output_frames'] == 81, 'Temporal geometry')
    # Small deterministic check of replacement bookkeeping and frame normalization.
    b = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 2, 2) / 32
    q = b + 0.25
    first, rest = b.clone(), b.clone()
    first[:, :, :1].copy_(q[:, :, :1]); rest[:, :, 1:].copy_(q[:, :, 1:])
    c = dict(teacher=b, native_terminal=q, first_only=first, rest_only=rest)
    identity = check_corner_identity(c, torch)
    curves = effect_curves(c, torch)
    require(curves['full']['mse_by_index'] == [0.0625]*3
            and curves['interaction']['mse_by_index'] == [0.]*3, 'FP64 slice reduction fixture')
    return dict(experiment='E045', status='complete', phase='CPU_readout_preparation',
                source=record(__file__), manifest=record(args.manifest), cuda_initialized=False,
                model_loads=0, generation_artifacts_read=False, fixture=identity,
                raw_layout='B,C,T,H,W; RGB FP32 pre-internal-clamp [-1,1] coordinates, possibly out of range',
                arithmetic='FP64 differences and squared means per time slice/frame; no gain ratio')


def plot(results, folder, plt):
    png, pdf = folder/'terminal_error_curves.png', folder/'terminal_error_curves.pdf'
    require(not png.exists() and not pdf.exists(), 'Preserve previous plots')
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.2), layout='constrained')
    colors = dict(full='#2369b0', first_only='#d47a15', rest_only='#248e65', interaction='#b54867')
    for row, case in enumerate(results.values()):
        ax = axes[row, 0]
        for name, label in (('cond', 'conditional'), ('uncond', 'unconditional'), ('cfg', 'CFG output')):
            ax.plot(range(21), case['prediction_error'][name]['mse_by_index'], label=label, linewidth=1.5)
        ax.set(title=f"seed {case['seed']}: actual prediction error", xlabel='Latent time index (0–20)',
               ylabel='Prediction MSE per element')
        ax = axes[row, 1]
        for name, label in (('diffusion_normalized', 'diffusion state'), ('vae_denormalized', 'VAE input (original std)')):
            ax.plot(range(21), case['latent_effects'][name]['full']['mse_by_index'], label=label, linewidth=1.5)
        ax.set(title='Terminal Q − B, separate coordinates', xlabel='Latent time index (0–20)',
               ylabel='Latent MSE per element')
        for col, space, title in ((2, 'pre_internal_clamp', 'RGB before internal clamp'),
                                  (3, 'post_internal_clamp', 'RGB after clamp [−1,1]')):
            ax = axes[row, col]
            for name in ('full', 'first_only', 'rest_only', 'interaction'):
                ax.plot(range(81), case['rgb_effects'][space][name]['mse_by_index'], label=name.replace('_', ' '),
                        color=colors[name], linestyle='--' if name == 'interaction' else '-', linewidth=1.4)
            ax.set(title=title, xlabel='Decoded frame index (0–80)', ylabel='RGB MSE per channel/pixel')
        for ax in axes[row]:
            ax.grid(alpha=.22); ax.legend(fontsize=7.5); ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.suptitle('E045 · same teacher terminal state · each seed reported separately\n'
                 'First/rest are latent replacements; interaction = Q − first − rest + B. No decoder-gain or quality estimate.', fontsize=11)
    fig.savefig(png, dpi=170); fig.savefig(pdf); plt.close(fig)
    return dict(png=record(png), pdf=record(pdf))


def summarize(args, torch, plt):
    plan = json.loads(args.manifest.read_text())
    references, cache = {}, {}

    def bind(entry):
        path = entry['file']
        if path not in cache:
            cache[path] = record(path)
        require(cache[path] == entry, 'Artifact changed: ' + path)
        references[path] = entry
        return Path(path)

    def load_json(path):
        entry = record(path); cache[entry['file']] = entry; references[entry['file']] = entry
        return json.loads(Path(path).read_text())

    def load_tensor(entry, shape, dtype):
        x = torch.load(bind(entry['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(isinstance(x, torch.Tensor) and list(x.shape) == shape and x.dtype == dtype, 'Saved tensor shape/dtype')
        signature = tensor_signature(x, torch)
        require(signature == entry['tensor'] and signature['finite'], 'Saved tensor contents')
        return x

    launch = load_json(args.report_dir/'launcher.json')
    require(launch['status'] == 'complete' and len(launch['workers']) == 2, 'Both workers must finish')
    index = {}
    for worker in launch['workers']:
        require(worker['status'] == 'complete' and worker['returncode'] == 0, 'Worker failed/incomplete')
        path = bind(worker['result'])
        report = json.loads(path.read_text())
        require(report['status'] == 'complete', 'Incomplete worker report')
        index[report['case']['case_id']] = report
    require(set(index) == {c['case_id'] for c in plan['cases']}, 'Fixed two seeds')
    result = dict(experiment='E045', status='running', cpu_only=True, source=record(__file__),
        manifest=record(args.manifest), cases={}, actual_totals={k: 0 for k in EXPECTED}, definitions=dict(
            latent_first='time index 0; MSE mean over batch/channel/spatial elements',
            latent_rest='arithmetic mean of the 20 per-slice MSEs at indices 1:21',
            rgb_first='decoded frame 0; MSE mean over batch/RGB channel/pixels',
            rgb_rest='arithmetic mean of the 80 per-frame MSEs at indices 1:81',
            full_effect='native_terminal − teacher', first_effect='first_only − teacher',
            rest_effect='rest_only − teacher', interaction='native_terminal − first_only − rest_only + teacher',
            rgb_coordinates='Raw decoder RGB before or after internal clamp[-1,1], before affine [0,1] conversion, uint8 rounding or video encoding'),
        limits=[
            'Both clapping seeds were selected after E044 visible defects; these are local diagnostic interventions, not held-out population evidence.',
            'Only the final same-state native prediction is intervened on. This does not identify effects of the full free-running quantized trajectory.',
            'UniPC final sigma goes to zero with lower_order_final; the final step reduces order. This is not evidence of multi-step history amplification.',
            'First latent produces one output frame and later latents four each, but causal decoder state couples their receptive fields. Output scheduling is not independence.',
            'All errors are per-element or per-frame means; no first-frame versus 80-frame summed-energy comparison is used.',
            'Latent and RGB coordinates differ. Their norm ratio is not a decoder gain, and no such ratio or quality score is reported.',
            'VAE denormalization is recomputed on CPU in the original FP32 operation order using original per-channel means/stds; actual intermediate GPU denormalized tensors were not saved.',
            'Clamped public RGB is reconstructed from saved preclamp RGB and checked against its captured tensor SHA; encoded media is not used for MSE.',
            'Teacher replay and BF16 shadow differences are numerical reproducibility descriptions, not quality/acceptance thresholds.'])
    shape = [1, 16, 21, 60, 104]
    for case in plan['cases']:
        tid = case['case_id']; worker = index[tid]
        require(worker['case'] == case and worker['settings'] == plan['settings'], 'Case/settings identity')
        require(worker['manifest'] == result['manifest'] and worker['actual_counts'] == EXPECTED, 'Manifest/actual calls')
        for key in EXPECTED:
            result['actual_totals'][key] += worker['actual_counts'][key]
        reference = load_json(bind(worker['reference_worker']))
        old = next(r for r in reference['cases'] if r['case_id'] == tid)
        require(worker['initial_noise'] == old['initial_noise'] and worker['embeddings'] == reference['embeddings'], 'Actual inherited input binding')
        capture = torch.load(bind(worker['terminal_capture']), map_location='cpu', weights_only=False, mmap=True)
        native = torch.load(bind(worker['native_outputs']['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(len(capture['calls']) == 2 and capture['history']['_step_index'] == 49, 'Actual terminal capture')
        require(capture['sample'].dtype == torch.float32 and list(capture['sample'].shape) == shape, 'Pre-step FP32 sample')
        teacher_predictions = dict(cond=capture['calls'][0]['output'], uncond=capture['calls'][1]['output'], cfg=capture['cfg_output'])
        for name in ('cond', 'uncond', 'cfg'):
            for tensor in (teacher_predictions[name], native[name]):
                require(tensor.dtype == torch.bfloat16 and list(tensor.shape) == shape
                        and bool(tensor.isfinite().all()), 'Actual BF16 prediction')
            sig = tensor_signature(native[name], torch)
            require(all(sig[k] == worker['native_outputs']['tensors'][name][k] for k in sig), 'Native prediction signature')
        cfg_replay = {}
        for name, payload in (('teacher', teacher_predictions), ('native', native)):
            reproduced = payload['uncond'] + 6. * (payload['cond'] - payload['uncond'])
            cfg_replay[name] = pair_curve(reproduced, payload['cfg'], torch)
        predictions = {name: pair_curve(native[name], teacher_predictions[name], torch) for name in ('cond', 'uncond', 'cfg')}
        corner_rows = {r['corner']: r for r in worker['corners']}
        require(len(worker['corners']) == len(corner_rows) == 4 and set(corner_rows) == set(CORNERS), 'All four corners')
        latents = {name: load_tensor(corner_rows[name]['latent'], shape, torch.float32) for name in CORNERS}
        identities = check_corner_identity(latents, torch)
        previous = load_tensor(worker['teacher_reference_final'], shape, torch.float32)
        require(worker['teacher_reference_final'] == old['final_latents'], 'E043 final identity')
        replay = pair_curve(latents['teacher'], previous, torch)
        shadow = load_tensor(worker['shadow_latent'], shape, torch.float32)
        shadow_error = pair_curve(shadow, latents['teacher'], torch)
        config = worker['actual_configs']['vae']
        means = torch.tensor(config['latents_mean'], dtype=torch.float32).view(1, 16, 1, 1, 1)
        stds = torch.tensor(config['latents_std'], dtype=torch.float32).view(1, 16, 1, 1, 1)
        require(bool((stds > 0).all()) and bool(stds.isfinite().all() & means.isfinite().all()), 'Original VAE normalization')
        inverse_std = 1.0 / stds
        denorm = {name: x / inverse_std + means for name, x in latents.items()}
        latent_effects = dict(diffusion_normalized=effect_curves(latents, torch), vae_denormalized=effect_curves(denorm, torch))
        raw, clamp_receipts = {}, {}
        for name in CORNERS:
            row = corner_rows[name]
            require(row['status'] == 'complete' and row['decoder_chunk_frames'] == [1]+[4]*20, 'Decoder corner/chunk completion')
            raw[name] = load_tensor(row['preclamp_raw'], [1, 3, 81, 480, 832], torch.float32)
            clamped = tensor_signature(raw[name], torch, clamp=True)
            require(clamped == row['public_raw'], 'Captured preclamp does not reproduce actual public clamp output')
            clamp_receipts[name] = dict(actual_public_signature_matches=True,
                public_raw=clamped, worker_comparison=row['clamp_replay'], video=row['video'],
                latent=row['latent'], preclamp_raw=row['preclamp_raw'])
        rgb = dict(pre_internal_clamp=effect_curves(raw, torch), post_internal_clamp=effect_curves(raw, torch, clamp=True))
        result['cases'][tid] = dict(**case, actual_counts=worker['actual_counts'],
            prediction_error=predictions, cfg_cpu_replay=cfg_replay, corner_exact_replacement=identities,
            latent_effects=latent_effects, rgb_effects=rgb, captured_clamp_consistency=clamp_receipts,
            vae_normalization=dict(mean=config['latents_mean'], std=config['latents_std'], operation='FP32 z / (1.0 / std) + mean'),
            teacher_vs_e043=replay, shadow_vs_teacher=shadow_error,
            worker_reproducibility=dict(teacher_vs_e043=worker['teacher_vs_e043'], shadow_vs_teacher=worker['shadow_vs_teacher']),
            terminal=dict(step_index=49, timestep=float(capture['timestep']),
                sigma=float(capture['history']['sigmas'][49]), next_sigma=float(capture['history']['sigmas'][50]),
                sample_signature=tensor_signature(capture['sample'], torch),
                captured_teacher_signatures=worker['terminal_signatures'], scheduler_config=capture['scheduler_config']))
        print(f'{tid}: FP64 latent/prediction slices and pre/post-clamp RGB frames complete', flush=True)
        del raw, latents, denorm, previous, shadow, capture, native, teacher_predictions
    figures = args.figure_dir or Path(plan['data_dir'])/'readout'
    result['actual_totals'].update(dit=result['actual_totals']['teacher_dit']+result['actual_totals']['native_dit'],
        scheduler=sum(result['actual_totals'][k] for k in ('teacher_scheduler', 'shadow_scheduler', 'candidate_scheduler')))
    result.update(status='complete', figures=plot(result['cases'], figures, plt), raw_file_bindings=references,
                  arithmetic='FP64 differences and squared means, one time slice/frame at a time; raw arrays mmap on CPU',
                  cuda_initialized=torch.cuda.is_initialized())
    require(result['cuda_initialized'] is False, 'CPU reducer initialized CUDA')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=PLAN)
    parser.add_argument('--report-dir', type=Path, default=RD)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--figure-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    output = args.output or args.report_dir / ('summary_check.json' if args.check_only else 'summary.json')
    require(not output.exists(), 'Preserve earlier result; use --output for a new attempt')
    result = dict(experiment='E045', status='failed_preserved', cpu_only=True)
    started = time.monotonic()
    try:
        torch, plt = load_cpu_libraries()
        result = check(args, torch) if args.check_only else summarize(args, torch, plt)
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['seconds'] = time.monotonic()-started
        save_json(output, result)
        print(json.dumps(dict(status=result['status'], output=str(output), sha256=record(output)['sha256'])), flush=True)


if __name__ == '__main__':
    main()
