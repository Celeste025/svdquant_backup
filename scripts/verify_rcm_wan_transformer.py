#!/usr/bin/env python3
"""Numerically compare an rCM Wan checkpoint with its Diffusers conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel
from rcm.networks.wan2pt1 import WanModel

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcm-ckpt", type=Path, required=True)
    parser.add_argument("--diffusers-pipeline", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=RESULTS_ROOT / "reports" / "rcm-wan-transformer-equivalence.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    raw = torch.load(args.rcm_ckpt, map_location="cpu", weights_only=True)
    state = {key.removeprefix("net."): value for key, value in raw.get("state_dict", raw).items()}
    state = {key: value for key, value in state.items() if not key.startswith("accum_")}
    native = WanModel(
        dim=1536, eps=1e-6, ffn_dim=8960, freq_dim=256, in_dim=16,
        model_type="t2v", num_heads=12, num_layers=30, out_dim=16, text_len=512,
    )
    missing, unexpected = native.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Native load mismatch: missing={missing}, unexpected={unexpected}")
    native = native.eval().to(device=device, dtype=torch.bfloat16)
    converted = WanTransformer3DModel.from_pretrained(
        args.diffusers_pipeline, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval().to(device)

    generator = torch.Generator(device=device).manual_seed(20260827)
    latent = torch.randn((1, 16, 5, 32, 48), generator=generator, device=device, dtype=torch.bfloat16)
    text = torch.randn((1, 512, 4096), generator=generator, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device)
    with torch.no_grad():
        native_out = native(x_B_C_T_H_W=latent, timesteps_B_T=timestep[:, None], crossattn_emb=text).float()
        converted_out = converted(
            hidden_states=latent, timestep=timestep, encoder_hidden_states=text, return_dict=False
        )[0].float()
    delta = native_out - converted_out
    report = {
        "shape": list(native_out.shape),
        "max_abs_error": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(native_out.flatten(), converted_out.flatten(), dim=0).item(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
