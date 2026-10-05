#!/usr/bin/env python3
"""E055: CPU-only H3 same-state output phase statistics and analytic head null.

No model forward, hidden-state inversion, sampling, or quality metric. Existing
E014 artifacts are read-only. Analysis requires a root-frozen manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from safetensors import safe_open

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/data1/models/svdquant-wjq/research/20261002/E014/evaluate')
CHECKPOINT = Path('/home/wjq/workspace/DiffSynth-Studio/models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors')
HEAD_KEY = 'final_layer.video_out.weight'
CASES = ('e009_p001_s00', 'e010_p030_s05', 'e010_p036_s14')
ARMS = ('bf16', 'plain', 'svd')
T, C, H, W, D = 37, 24, 36, 64, 5376
PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
LAGS = ((0, 1), (1, 0), (1, 1))
SUPPORT = [0, 34, 0, 62]
GEOMETRY = [T, C, H, W]
CENTERING = 'per-time-per-channel spatial mean'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_check(path):
    doc = json.loads(path.read_text())
    expected = dict(experiment='E055', script_sha256=sha(__file__),
                    expected_geometry=GEOMETRY, centering=CENTERING,
                    lags=[list(x) for x in LAGS], common_origin_support=SUPPORT)
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ValueError(f'Manifest mismatch {key}: {doc.get(key)!r} != {value!r}')
    plan = doc['plan']
    if sha(plan['path']) != plan['sha256']:
        raise ValueError('Frozen plan SHA mismatch')
    return doc


def load_case(case):
    payloads = {}
    for arm in ARMS:
        path = BASE / arm / (case + '.pt')
        value = torch.load(path, map_location='cpu', weights_only=False)
        raw = value['raw_outputs']['video']
        if raw.shape != (T * (H//2) * (W//2), C*4):
            raise ValueError(f'Bad output shape: {path}: {raw.shape}')
        if raw.dtype != torch.bfloat16 or not torch.isfinite(raw).all():
            raise ValueError(f'Invalid output dtype/finite: {path}')
        payloads[arm] = value
    if not (payloads['bf16']['actual_dit_inputs'] == payloads['plain']['actual_dit_inputs']
            == payloads['svd']['actual_dit_inputs']):
        raise ValueError(f'Actual input signatures differ for {case}')
    return payloads


def unpatchify(raw):
    # Real H3 channel-major output row index is 4*c+2*p_h+p_w.
    return (raw.reshape(T, H//2, W//2, C, 2, 2)
            .permute(0, 3, 1, 4, 2, 5).reshape(T, C, H, W))


def phase_mean(field):
    return torch.stack([field[:, :, p::2, q::2].mean((-2, -1))
                        for p, q in PHASES], -1)


def covariance(field, lag):
    dh, dw = lag
    # Identical even-sized source support for all phases and lags, including 0.
    return torch.stack([
        (field[:, :, p:34:2, q:62:2]
         * field[:, :, p+dh:34+dh:2, q+dw:62+dw:2]).mean((-2, -1))
        for p, q in PHASES], -1)


def head_expectation(weight):
    """Exact finite-grid centered covariance under iid isotropic hidden noise.

    Unit hidden variance. y[n,c,p]=W[c,p] eps[n]; independent token eps.
    Spatial mean removal gives:
      1[n=m] <Wp,Wq> - (<Wp,S>+<Wq,S>)/(4M) + <S,S>/(16M).
    M=(H/2)*(W/2); S=sum_p Wp. Shared-channel covariances are unused.
    """
    w = weight.reshape(C, 4, D)
    gram = torch.einsum('cpd,cqd->cpq', w, w)
    sums = gram.sum(-1)
    allsum = gram.sum((-2, -1))
    m = (H//2) * (W//2)
    out = {}
    for dh, dw in ((0, 0), *LAGS):
        values = []
        for p, q in PHASES:
            src = 2*p + q
            dst = 2*((p+dh)%2) + ((q+dw)%2)
            same_token = (p+dh)//2 == 0 and (q+dw)//2 == 0
            values.append((gram[:, src, dst] if same_token else torch.zeros(C, dtype=torch.float64))
                          - (sums[:, src]+sums[:, dst])/(4*m) + allsum/(16*m))
        out[f'{dh},{dw}'] = torch.stack(values, -1)
    expected_energy = out['0,0'].mean()
    if expected_energy <= 0:
        raise ValueError('Degenerate head null')
    return out, expected_energy


def compare_summary(observed_normalized, expected_normalized):
    # Descriptive, no fitted gain, hypothesis threshold, or automatic decision.
    o = observed_normalized - observed_normalized.mean(-1, keepdim=True)
    n = expected_normalized - expected_normalized.mean(-1, keepdim=True)
    n = n.unsqueeze(0).expand_as(o)
    onorm, nnorm = o.square().sum().sqrt(), n.square().sum().sqrt()
    tiny = torch.finfo(torch.float64).tiny
    return dict(observed_phase_contrast_rms=float(o.square().mean().sqrt()),
                head_null_phase_contrast_rms=float(n.square().mean().sqrt()),
                mismatch_over_observed_contrast_norm=float((o-n).square().sum().sqrt()/onorm.clamp_min(tiny)),
                phase_contrast_cosine=float((o*n).sum()/(onorm*nnorm).clamp_min(tiny)),
                normalized_covariance_mean_by_phase=observed_normalized.mean((0, 1)).tolist(),
                head_null_normalized_covariance_mean_by_phase=expected_normalized.mean(0).tolist())


def describe_case(case, payloads):
    return dict(case_id=case, same_actual_input_signatures=True,
                artifacts={a: dict(path=str(BASE/a/(case+'.pt')),
                                   bytes=(BASE/a/(case+'.pt')).stat().st_size,
                                   raw_video_shape=list(payloads[a]['raw_outputs']['video'].shape),
                                   raw_video_dtype=str(payloads[a]['raw_outputs']['video'].dtype)) for a in ARMS})


def analyze_case(case, payloads, unit_cov, null_energy):
    result = describe_case(case, payloads)
    reference = payloads['bf16']['raw_outputs']['video'].double()
    arms = {}
    for arm in ('plain', 'svd'):
        error = unpatchify(payloads[arm]['raw_outputs']['video'].double() - reference)
        spatial_mean = error.mean((-2, -1), keepdim=True)
        centered = error - spatial_mean
        energy = centered.square().mean((1, 2, 3))
        if (energy <= 0).any():
            raise ValueError(f'Zero error energy {case}/{arm}')
        scale = energy[:, None, None]
        lags = {}
        for lag in ((0, 0), *LAGS):
            key = f'{lag[0]},{lag[1]}'
            cov = covariance(centered, lag)
            normalized = cov / scale
            expected_normalized = unit_cov[key] / null_energy
            lags[key] = dict(covariance=cov.tolist(), normalized_covariance=normalized.tolist(),
                             summary=compare_summary(normalized, expected_normalized))
        arms[arm] = dict(error_direction='quantized raw output minus BF16 raw output; velocity has opposite sign',
                         raw_error_mean_square_by_time=error.square().mean((1, 2, 3)).tolist(),
                         centered_error_mean_square_by_time=energy.tolist(),
                         matched_hidden_noise_variance_by_time=(energy/null_energy).tolist(),
                         spatial_mean_by_time_channel=spatial_mean.squeeze(-1).squeeze(-1).tolist(),
                         raw_phase_mean=phase_mean(error).tolist(),
                         centered_phase_mean=phase_mean(centered).tolist(), lags=lags)
    result['arms'] = arms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('check', 'analyze'), required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    if args.phase == 'analyze' and args.manifest is None:
        raise ValueError('Analysis requires a frozen manifest')
    if args.manifest is not None:
        manifest_check(args.manifest)
    historical = json.loads((ROOT/'results/research/E014/E014_evaluate_bf16.json').read_text())
    inv = historical['resident_conversion']['inventory_after'][HEAD_KEY]
    if inv['dtype'] != 'torch.bfloat16' or inv['shape'] != [96, D]:
        raise ValueError('Historical actual head precision/shape mismatch')
    with safe_open(str(CHECKPOINT), framework='pt', device='cpu') as handle:
        raw_head = handle.get_tensor(HEAD_KEY)
    if raw_head.shape != (96, D) or raw_head.dtype != torch.float32:
        raise ValueError('Unexpected stored head')
    head = raw_head.to(torch.bfloat16).double()
    result = dict(experiment='E055', phase=args.phase, status='complete',
                  script_sha256=sha(__file__), manifest_path=str(args.manifest) if args.manifest else None,
                  checkpoint=dict(path=str(CHECKPOINT), key=HEAD_KEY, stored_dtype='torch.float32',
                                  actual_computation_dtype='torch.bfloat16', shape=[96,D],
                                  checkpoint_bytes=CHECKPOINT.stat().st_size,
                                  small_bf16_head_sha256=hashlib.sha256(raw_head.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()).hexdigest()),
                  geometry=dict(error_tensor_axes=['latent_time','channel','latent_h','latent_w'],shape=GEOMETRY,
                                output_row_index='4*channel+2*phase_h+phase_w', phases=[list(x) for x in PHASES],
                                common_origin_support_exclusive=SUPPORT, source_pairs_per_phase=17*31),
                  centering=CENTERING,
                  null=dict(model='iid zero-mean isotropic noise immediately before actual BF16 video_out',
                            computation='FP64 analytic real-linear head covariance; includes finite-grid spatial-mean subtraction',
                            limitations=['Does not model BF16 output rounding or nonlinear RMSNorm/modulation before the perturbation site',
                                         'Does not exclude colored or channel-anisotropic hidden error',
                                         'No hidden residual captured or reconstructed; no quality or novelty claim']),
                  cases=[])
    if args.phase == 'analyze':
        unit_cov, null_energy = head_expectation(head)
        result['null']['unit_variance_mean_centered_output_energy'] = float(null_energy)
        result['null']['unit_variance_covariance_by_channel_phase'] = {k:v.tolist() for k,v in unit_cov.items()}
        result['null']['normalized_covariance_by_channel_phase'] = {k:(v/null_energy).tolist() for k,v in unit_cov.items()}
        result['null']['normalization'] = 'One variance per actual case/arm/time matches full-grid, all-channel centered mean-square error; no fitted phase/channel gains'
    for case in CASES:
        payloads = load_case(case)
        result['cases'].append(analyze_case(case, payloads, unit_cov, null_energy)
                               if args.phase == 'analyze' else describe_case(case, payloads))
    result.update(cuda_initialized=torch.cuda.is_initialized(), gpu_calls=0, model_forwards=0,
                  seconds=time.monotonic()-started)
    if result['cuda_initialized']:
        raise RuntimeError('Unexpected CUDA initialization')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status='complete',phase=args.phase,output=str(args.output),
                          seconds=result['seconds'],cuda_initialized=False)))


if __name__ == '__main__':
    main()
