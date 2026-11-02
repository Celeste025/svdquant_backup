#!/usr/bin/env python3
"""E103: reduce the full paired DiT tensors without discarding their structure."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel().copy()
    y = np.asarray(b, dtype=np.float64).ravel().copy()
    x -= x.mean()
    y -= y.mean()
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float((x @ y) / den) if den > 0 else 0.0


def top_share(values: np.ndarray, fraction: float) -> tuple[float, np.ndarray]:
    flat = values.ravel()
    count = max(1, math.ceil(len(flat) * fraction))
    top = np.argpartition(flat, -count)[-count:]
    return float(flat[top].sum(dtype=np.float64) / flat.sum(dtype=np.float64)), top


def vector_cos(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    num = float(np.sum(x * y, dtype=np.float64))
    den = math.sqrt(float(np.sum(x * x, dtype=np.float64)) *
                    float(np.sum(y * y, dtype=np.float64)))
    return num / den if den > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--arrays", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["format"] != "h3-e103-paired-output-manifest-v1" or len(manifest["records"]) != 20:
        raise RuntimeError("incomplete paired raw capture")
    t, h, w = manifest["video_grid_thw"]
    rng = np.random.default_rng(103)
    step_rows = []
    channel_e, channel_s, frame_e, frame_s = [], [], [], []
    audio_channel_e, audio_channel_s = [], []
    token_e_all, token_s_all = [], []
    selected_maps = {}
    selected_hist = {}
    selected_signed_midframe = {}
    for record in manifest["records"]:
        step = int(record["step"])
        payload = torch.load(record["path"], map_location="cpu", weights_only=False)
        if payload["step"] != step or tuple(payload["video_grid_thw"]) != (t, h, w):
            raise RuntimeError(f"step {step}: raw metadata mismatch")
        ref = payload["video_bf16"].float().numpy().reshape(t, h, w, 96)
        err = payload["video_error_fp32"].numpy().reshape(t, h, w, 96)
        quant = payload["video_svdquant"].float().numpy().reshape(t, h, w, 96)
        if not np.array_equal(err, quant - ref) or not np.isfinite(err).all():
            raise RuntimeError(f"step {step}: stored difference or finite check failed")
        e2 = np.square(err, dtype=np.float64)
        s2 = np.square(ref, dtype=np.float64)
        token_e = e2.sum(axis=-1)
        token_s = s2.sum(axis=-1)
        ce = e2.sum(axis=(0, 1, 2))
        cs = s2.sum(axis=(0, 1, 2))
        fe = token_e.sum(axis=(1, 2))
        fs = token_s.sum(axis=(1, 2))
        channel_e.append(ce)
        channel_s.append(cs)
        frame_e.append(fe)
        frame_s.append(fs)
        token_e_all.append(token_e.astype(np.float32))
        token_s_all.append(token_s.astype(np.float32))
        audio_ref = payload["audio_bf16"].float().numpy()
        audio_err = payload["audio_error_fp32"].numpy()
        audio_channel_e.append(np.square(audio_err, dtype=np.float64).sum(axis=0))
        audio_channel_s.append(np.square(audio_ref, dtype=np.float64).sum(axis=0))
        nmse = float(token_e.sum() / token_s.sum())
        if abs(nmse - payload["metrics"]["video"]["nmse"]) > 1e-7:
            raise RuntimeError(f"step {step}: NMSE did not reproduce the capture")
        token_top_share, top_ids = top_share(token_e, 0.01)
        paired_signal_share = float(token_s.ravel()[top_ids].sum() / token_s.sum())
        abs_err = np.abs(err).ravel()
        quantiles = np.quantile(abs_err, [0.5, 0.9, 0.99, 0.999]).tolist()
        element_top_share, _ = top_share(e2, 0.001)
        # Remove frame-level magnitude shifts before measuring local clustering.
        magnitude = np.sqrt(token_e / 96)
        logmag = np.log(np.maximum(magnitude, 1e-12))
        logmag = logmag - logmag.mean(axis=(1, 2), keepdims=True)
        ref_logmag = np.log(np.maximum(np.sqrt(token_s / 96), 1e-12))
        ref_logmag = ref_logmag - ref_logmag.mean(axis=(1, 2), keepdims=True)
        shuffled = np.stack([frame.ravel()[rng.permutation(h * w)].reshape(h, w)
                             for frame in logmag])
        # Signed-error alignment tests coherent directions, beyond magnitude hot spots.
        centered_error = err - err.mean(axis=(1, 2), keepdims=True)
        shuffled_error = np.stack([frame.reshape(h * w, 96)[rng.permutation(h * w)].reshape(h, w, 96)
                                   for frame in centered_error])
        row = {
            "step": step, "unique_timesteps": record["unique_timesteps"],
            "video_nmse": nmse, "audio_nmse": payload["metrics"]["audio"]["nmse"],
            "video_ref_rms": float(np.sqrt(s2.mean())),
            "video_error_rms": float(np.sqrt(e2.mean())),
            "video_error_mean": float(err.mean()),
            "abs_error_quantiles": {name: value for name, value in zip(
                ("p50", "p90", "p99", "p999"), quantiles, strict=True)},
            "top_1pct_token_error_energy_share": token_top_share,
            "signal_energy_share_at_same_top_tokens": paired_signal_share,
            "top_0p1pct_element_error_energy_share": element_top_share,
            "spatial_h_magnitude_corr_frame_demeaned": corr(logmag[:, 1:, :], logmag[:, :-1, :]),
            "spatial_w_magnitude_corr_frame_demeaned": corr(logmag[:, :, 1:], logmag[:, :, :-1]),
            "temporal_magnitude_corr_frame_demeaned": corr(logmag[1:], logmag[:-1]),
            "reference_spatial_h_magnitude_corr_frame_demeaned": corr(ref_logmag[:, 1:, :], ref_logmag[:, :-1, :]),
            "reference_spatial_w_magnitude_corr_frame_demeaned": corr(ref_logmag[:, :, 1:], ref_logmag[:, :, :-1]),
            "reference_temporal_magnitude_corr_frame_demeaned": corr(ref_logmag[1:], ref_logmag[:-1]),
            "shuffled_spatial_h_corr": corr(shuffled[:, 1:, :], shuffled[:, :-1, :]),
            "shuffled_spatial_w_corr": corr(shuffled[:, :, 1:], shuffled[:, :, :-1]),
            "shuffled_temporal_corr": corr(shuffled[1:], shuffled[:-1]),
            "signed_error_neighbor_cosine_h": vector_cos(centered_error[:, 1:, :, :], centered_error[:, :-1, :, :]),
            "signed_error_neighbor_cosine_w": vector_cos(centered_error[:, :, 1:, :], centered_error[:, :, :-1, :]),
            "signed_error_neighbor_cosine_t": vector_cos(centered_error[1:], centered_error[:-1]),
            "shuffled_signed_error_cosine_h": vector_cos(shuffled_error[:, 1:, :, :], shuffled_error[:, :-1, :, :]),
            "shuffled_signed_error_cosine_w": vector_cos(shuffled_error[:, :, 1:, :], shuffled_error[:, :, :-1, :]),
            "shuffled_signed_error_cosine_t": vector_cos(shuffled_error[1:], shuffled_error[:-1]),
        }
        step_rows.append(row)
        if step in (0, 10, 19):
            selected_maps[f"step{step:02d}_token_error_rms"] = magnitude.astype(np.float32)
            selected_maps[f"step{step:02d}_token_ref_rms"] = np.sqrt(token_s / 96).astype(np.float32)
            selected_signed_midframe[step] = err[18].copy()
            sample = abs_err[rng.choice(abs_err.size, min(150000, abs_err.size), replace=False)]
            selected_hist[f"step{step:02d}_abs_error_over_ref_rms"] = (sample / row["video_ref_rms"]).astype(np.float32)
        print(f"step {step:02d}: NMSE {nmse:.6f}, top1% token error share {token_top_share:.3f}, "
              f"neighbor h/w/t {row['spatial_h_magnitude_corr_frame_demeaned']:.3f}/"
              f"{row['spatial_w_magnitude_corr_frame_demeaned']:.3f}/"
              f"{row['temporal_magnitude_corr_frame_demeaned']:.3f}", flush=True)
    ce_all = np.stack(channel_e)
    cs_all = np.stack(channel_s)
    fe_all = np.stack(frame_e)
    fs_all = np.stack(frame_s)
    ace_all = np.stack(audio_channel_e)
    acs_all = np.stack(audio_channel_s)
    token_e_stack = np.stack(token_e_all)
    token_s_stack = np.stack(token_s_all)
    pooled_ce = ce_all.sum(axis=0)
    pooled_cs = cs_all.sum(axis=0)
    pooled_ace = ace_all.sum(axis=0)
    pooled_acs = acs_all.sum(axis=0)
    top_channels = np.argsort(pooled_ce)[::-1][:12]
    top_channel = int(top_channels[0])
    channel_shares = ce_all / ce_all.sum(axis=1, keepdims=True)
    adjacent_channel_pattern_corr = [corr(channel_shares[i], channel_shares[i + 1])
                                     for i in range(19)]
    adjacent_token_pattern_corr = [corr(np.log(np.maximum(token_e_stack[i], 1e-30)),
                                        np.log(np.maximum(token_e_stack[i + 1], 1e-30)))
                                   for i in range(19)]
    for step, frame in selected_signed_midframe.items():
        selected_maps[f"step{step:02d}_signed_error_top_channel_frame18"] = frame[:, :, top_channel]
    summary = {
        "format": "h3-e103-output-error-pattern-summary-v1",
        "scope": f"prompt {manifest['prompt_id']}, seed {manifest['seed']}, 20 teacher-forced BF16 trajectory DiT calls, rank-32 SVDQuant dynamic",
        "manifest": str(args.manifest.resolve()),
        "video_grid_thw": [t, h, w], "video_channels": 96,
        "metric_definitions": {
            "nmse": "sum(quant-BF16)^2 / sum(BF16)^2 over the selected video elements",
            "top_1pct_token_error_energy_share": "fraction of total squared error in the 1% highest-error video tokens",
            "adjacency_corr": "Pearson correlation of log(token error RMS), demeaned separately within each latent frame; adjacent axes t/h/w; permutation baseline shuffles spatial positions independently within each frame",
            "signed_error_neighbor_cosine": "global cosine of adjacent 96-dimensional signed error vectors after subtracting each latent frame's per-channel mean error; spatial-position permutation baseline",
        },
        "pooled_video_nmse": float(ce_all.sum() / cs_all.sum()),
        "top_12_channels_by_error_energy": [
            {"channel": int(i), "error_energy_share": float(pooled_ce[i] / pooled_ce.sum()),
             "signal_energy_share": float(pooled_cs[i] / pooled_cs.sum()),
             "channel_nmse": float(pooled_ce[i] / pooled_cs[i])}
            for i in top_channels],
        "signed_spatial_map_top_channel": top_channel,
        "top_10_channels_error_share": float(pooled_ce[np.argsort(pooled_ce)[-10:]].sum() / pooled_ce.sum()),
        "top_10_channels_signal_share_at_same_channels": float(pooled_cs[np.argsort(pooled_ce)[-10:]].sum() / pooled_cs.sum()),
        "adjacent_step_channel_error_share_corr": adjacent_channel_pattern_corr,
        "adjacent_step_token_error_log_energy_corr": adjacent_token_pattern_corr,
        "subpatch_2x2": {
            "error_energy_share": (pooled_ce.reshape(24, 2, 2).sum(axis=0) / pooled_ce.sum()).tolist(),
            "signal_energy_share": (pooled_cs.reshape(24, 2, 2).sum(axis=0) / pooled_cs.sum()).tolist(),
            "nmse": (pooled_ce.reshape(24, 2, 2).sum(axis=0) /
                     pooled_cs.reshape(24, 2, 2).sum(axis=0)).tolist(),
        },
        "pooled_audio_nmse": float(ace_all.sum() / acs_all.sum()),
        "top_8_audio_channels_by_error_energy": [
            {"channel": int(i), "error_energy_share": float(pooled_ace[i] / pooled_ace.sum()),
             "signal_energy_share": float(pooled_acs[i] / pooled_acs.sum()),
             "channel_nmse": float(pooled_ace[i] / pooled_acs[i])}
            for i in np.argsort(pooled_ace)[::-1][:8]],
        "steps": step_rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(args.arrays, channel_error_energy=ce_all, channel_signal_energy=cs_all,
                        frame_error_energy=fe_all, frame_signal_energy=fs_all,
                        token_error_energy=token_e_stack, token_signal_energy=token_s_stack,
                        audio_channel_error_energy=ace_all, audio_channel_signal_energy=acs_all,
                        **selected_maps, **selected_hist)
    print(f"COMPLETE {args.summary} and {args.arrays}", flush=True)


if __name__ == "__main__":
    main()
