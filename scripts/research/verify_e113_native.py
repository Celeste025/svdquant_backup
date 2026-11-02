#!/usr/bin/env python3
"""Independent completion audit for E113 native H3 mixed-attention videos."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av

from summarize_e113_native_videos import ROOT, sha256

EXPECTED = {'race': (16, 20026), 'clap0': (161, 38001610)}
ARMS = ('sage3', 'native_fp4', 'native_mixed')
PAIRS = ('sage3_vs_bf16', 'native_fp4_vs_bf16',
         'native_mixed_vs_bf16', 'native_mixed_vs_native_fp4')


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def valid_media(path: Path, *, audio: bool, height: int = 576) -> None:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        if (video.width, video.height, float(video.average_rate)) != (1024, height, 24.0):
            raise RuntimeError(f'video format mismatch: {path}')
        if audio and len(container.streams.audio) != 1:
            raise RuntimeError(f'audio stream mismatch: {path}')
        if sum(1 for _ in container.decode(video=0)) != 124:
            raise RuntimeError(f'frame count mismatch: {path}')


def audit_case(case: str, *, require_analysis: bool) -> dict:
    directory = ROOT / 'results/research/E113' / case
    prompt, seed = EXPECTED[case]
    reports = [read(directory / f'{arm}_denoise.json') for arm in ARMS]
    media = [read(directory / f'{arm}_video.json') for arm in ARMS]
    identities = set()
    for arm, denoise, video in zip(ARMS, reports, media, strict=True):
        if (denoise['case'], denoise['arm'], denoise['prompt_id'], denoise['seed'],
            denoise['status']) != (case, arm, prompt, seed, 'denoise_complete'):
            raise RuntimeError(f'wrong denoise identity/status: {case}/{arm}')
        if [row['step'] for row in denoise['steps']] != list(range(20)):
            raise RuntimeError(f'missing or duplicated step: {case}/{arm}')
        for row in denoise['steps']:
            if row['attention_main_calls'] != 50 or row['attention_refiner_calls'] != 2:
                raise RuntimeError(f'attention call count changed: {case}/{arm}')
            if not all(math.isfinite(row[key]) and row[key] > 0 for key in
                       ('dit_seconds', 'video_output_energy', 'audio_output_energy')):
                raise RuntimeError(f'nonfinite step metric: {case}/{arm}')
        if sha256(Path(denoise['terminal_latents'])) != denoise['terminal_sha256']:
            raise RuntimeError(f'terminal latent hash mismatch: {case}/{arm}')
        identities.add((json.dumps(denoise['settings'], sort_keys=True),
                        denoise['initial_latent_sha256']['video'],
                        denoise['initial_latent_sha256']['audio']))
        path = Path(video['video'])
        if ((video['case'], video['arm'], video['frames'], video['audio_streams']) !=
                (case, arm, 124, 1)):
            raise RuntimeError(f'wrong video manifest: {case}/{arm}')
        if sha256(path) != video['sha256']:
            raise RuntimeError(f'video hash mismatch: {case}/{arm}')
        if sha256(directory / f'{arm}_denoise.json') != video['denoise_report_sha256']:
            raise RuntimeError(f'denoise report hash mismatch: {case}/{arm}')
        valid_media(path, audio=True)
    if len(identities) != 1:
        raise RuntimeError(f'arm settings or initial latents differ: {case}')
    result = {'case': case, 'prompt_id': prompt, 'seed': seed,
              'arms': list(ARMS), 'steps_each': 20, 'frames_each': 124,
              'same_settings_and_initial_latents': True}
    if case == 'clap0':
        historical = read(ROOT / 'results/research/E073/denoise_full_bf16_r0.json')
        bf16 = next(row for row in historical['cases'] if row['case_id'] == 'vbench0161_r0')
        if (bf16['prompt_id'], bf16['seed']) != (prompt, seed):
            raise RuntimeError('historical BF16 clap identity differs')
        observed = reports[0]['initial_latent_sha256']
        expected = bf16['initial_noise_sha256']
        if (observed['video'], observed['audio']) != (expected['video_latents'],
                                                       expected['audio_latents']):
            raise RuntimeError('historical BF16 clap initial noise differs')
        result['bf16_initial_latents_sha256_match'] = True
    if not require_analysis:
        return result

    summary_path = directory / 'four_arm_attention_summary.json'
    summary = read(summary_path)
    lpips = read(directory / 'four_arm_attention_lpips.json')
    if summary['case'] != case or summary['frames'] != 124:
        raise RuntimeError(f'invalid RGB summary: {case}')
    for arm, video in zip(ARMS, media, strict=True):
        if summary['sha256'][arm] != video['sha256']:
            raise RuntimeError(f'summary source mismatch: {case}/{arm}')
    bf16 = Path(summary['sources']['bf16'])
    if sha256(bf16) != summary['sha256']['bf16']:
        raise RuntimeError(f'BF16 hash mismatch: {case}')
    valid_media(bf16, audio=False)
    for key in ('montage', 'poster'):
        if sha256(Path(summary[key])) != summary[f'{key}_sha256']:
            raise RuntimeError(f'{key} hash mismatch: {case}')
    valid_media(Path(summary['montage']), audio=False, height=636)
    for pair in PAIRS:
        value = summary['rgb_nmse'][pair]
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f'RGB metric invalid: {case}/{pair}')
    if ((lpips['case'], lpips['frames'], lpips['source_summary_sha256']) !=
            (case, 124, sha256(summary_path))):
        raise RuntimeError(f'LPIPS source mismatch: {case}')
    for pair in PAIRS:
        values = lpips['per_frame'][pair]
        if len(values) != 124 or not all(math.isfinite(x) and x >= 0 for x in values):
            raise RuntimeError(f'LPIPS frame metric invalid: {case}/{pair}')
        if abs(sum(values) / 124 - lpips['pairs'][pair]) > 1e-8:
            raise RuntimeError(f'LPIPS mean mismatch: {case}/{pair}')
    result.update(rgb_nmse={key: summary['rgb_nmse'][key] for key in PAIRS},
                  lpips={key: lpips['pairs'][key] for key in PAIRS},
                  lpips_model_sha256=lpips['metric_state_sha256'],
                  montage=summary['montage'])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=(*EXPECTED, 'all'), default='all')
    parser.add_argument('--require-analysis', action='store_true')
    args = parser.parse_args()
    numerical = read(ROOT / 'results/research/E113/native_vs_pytorch_group128_full.json')
    if (numerical['fp4_group'], numerical['fp8_group']) != (16, 128):
        raise RuntimeError('numerical validation group sizes changed')
    if not numerical['results'] or not all(
        row['finite'] and row['nmse_vs_pytorch'] < 1e-5 and
        all(row['mma_ptx'].values()) for row in numerical['results']
    ):
        raise RuntimeError('Triton/PyTorch native MMA gate failed')
    cases = EXPECTED if args.case == 'all' else (args.case,)
    audited = [audit_case(case, require_analysis=args.require_analysis) for case in cases]
    if args.require_analysis and len(audited) == 2 and len({row['lpips_model_sha256'] for row in audited}) != 1:
        raise RuntimeError('LPIPS models differ across cases')
    print(json.dumps({'status': 'pass', 'numerical_rows': len(numerical['results']),
                      'cases': audited}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
