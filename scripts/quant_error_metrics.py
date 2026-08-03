#!/usr/bin/env python3
"""Shared MSE / NMSE helpers for SVDQuant diagnostic experiments."""

from __future__ import annotations

import torch


def mse_nmse(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> dict[str, float]:
    """MSE = mean((est-ref)^2); NMSE = MSE / mean(ref^2)."""
    ref = reference.detach().float().reshape(-1)
    est = estimate.detach().float().reshape(-1)
    error = est - ref
    mse = error.square().mean()
    power = ref.square().mean().clamp_min(eps)
    return {
        "mse": float(mse),
        "nmse": float(mse / power),
        "nrmse": float((mse / power).sqrt()),
        "reference_rms": float(power.sqrt()),
        "error_rms": float(mse.sqrt()),
    }


def quant_metrics(x: torch.Tensor, smooth: torch.Tensor, unsigned: bool) -> dict:
    """Post-smooth dynamic group-64 INT4 reconstruction error on activations."""
    # Keep metrics on CPU to avoid competing with the diffusion model for VRAM.
    x = x.detach().float().cpu()
    xs = x / smooth.detach().float().cpu()
    groups = xs.reshape(-1, xs.shape[-1] // 64, 64)
    if unsigned:
        scale = groups.amax(-1, keepdim=True).clamp_min(1e-6) / 15
        q = (groups / scale).round().clamp(0, 15) * scale
    else:
        scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 7
        q = (groups / scale).round().clamp(-7, 7) * scale
    err = q - groups
    power = groups.square().mean().clamp_min(1e-12)
    abs_flat = groups.abs().flatten()
    quantile_values = abs_flat
    if quantile_values.numel() > 1_000_000:
        stride = (quantile_values.numel() + 999_999) // 1_000_000
        quantile_values = quantile_values[::stride]
    rms = power.sqrt()
    return {
        "mse": err.square().mean().item(),
        "nmse": (err.square().mean() / power).item(),
        "mae": err.abs().mean().item(),
        "nmae": (err.abs().mean() / groups.abs().mean().clamp_min(1e-12)).item(),
        "max_over_rms": (abs_flat.max() / rms).item(),
        "p999_over_rms": (torch.quantile(quantile_values, 0.999) / rms).item(),
        "p99_over_rms": (torch.quantile(quantile_values, 0.99) / rms).item(),
        "tail_gt_6rms": (abs_flat > 6 * rms).float().mean().item(),
        "rms": rms.item(),
        "absmax": abs_flat.max().item(),
    }
