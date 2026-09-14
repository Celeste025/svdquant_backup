#!/usr/bin/env python3
"""Paired four-step rCM rollout metrics for the saved fastmotion case."""
from __future__ import annotations

import csv
import gc
import json
import math
from pathlib import Path

import torch
from diffusers import WanPipeline, WanTransformer3DModel

from exp_rcm_blockwise_lowrank_pilot import freeze_all, install_activation_ste, restore_activation_processors
from exp_rcm_joint_smooth_lowrank import CKPT, MODEL, RCM_TRANSFORMER, DynamicSmooth
from infer_rcm_joint_full20 import load_joint_state, load_quantized_transformer

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/reports/rcm_joint_smooth_lowrank/full20_20260909/epoch_10.pt"
OUT = ROOT / "results/reports/rcm_joint_full20_unseen/fastmotion_seed308_epoch10_per_step"
PROMPT = "A professional skateboarder performing a high-speed kickflip down a concrete stair set, dynamic tracking shot."

def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Return raw MSE and reference energy before forming NMSE."""
    mse = float((a.float() - b.float()).square().mean().item())
    ref_mse = float(b.float().square().mean().item())
    return {"mse": mse, "reference_energy": ref_mse,
            "nmse": mse / max(ref_mse, 1e-12)}

def main() -> None:
    device = torch.device("cuda")
    # This follows infer_rcm_joint_full20.py exactly through its sampler setup.
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    pipe.to(device)
    load_quantized_transformer(pipe, CKPT, MODEL)
    student = pipe.transformer
    freeze_all(student)
    teacher = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16).to(device)
    freeze_all(teacher)
    dynamic = DynamicSmooth(student, teacher, CKPT, bound=1.25).to(device)
    originals, _ = install_activation_ste(student)
    load_joint_state(student, dynamic, STATE)

    # Text/VAE are only needed before the numerical rollout.  Keep GPU headroom for both DiTs.
    prompt_embeds, _ = pipe.encode_prompt(PROMPT, do_classifier_free_guidance=False,
        max_sequence_length=512, device=device, dtype=student.dtype)
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(308)
    initial = pipe.prepare_latents(batch_size=1, num_channels_latents=student.config.in_channels,
        height=480, width=832, num_frames=81, dtype=torch.float32, device=device, generator=generator)
    trig = torch.tensor([math.atan(80.0), 1.5, 1.4, 1.0, 0.0], dtype=torch.float64, device=device)
    ts = torch.sin(trig) / (torch.cos(trig) + torch.sin(trig))
    bf_latent = initial.to(torch.float64) * ts[0]
    q_latent = bf_latent.clone()
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    rows = []
    with torch.inference_mode():
        for step, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
            timestep = (t_cur.float() * ones * 1000).to(dtype=student.dtype)
            # teacher-forced: both see BF16 reference trajectory; rollout: both see quant trajectory.
            bf_v = teacher(hidden_states=bf_latent.to(teacher.dtype), timestep=timestep,
                encoder_hidden_states=prompt_embeds, return_dict=False)[0].to(torch.float64)
            tf_q_v = student(hidden_states=bf_latent.to(student.dtype), timestep=timestep,
                encoder_hidden_states=prompt_embeds, return_dict=False)[0].to(torch.float64)
            rollout_bf_v = teacher(hidden_states=q_latent.to(teacher.dtype), timestep=timestep,
                encoder_hidden_states=prompt_embeds, return_dict=False)[0].to(torch.float64)
            q_v = student(hidden_states=q_latent.to(student.dtype), timestep=timestep,
                encoder_hidden_states=prompt_embeds, return_dict=False)[0].to(torch.float64)
            noise = torch.randn(q_latent.shape, dtype=torch.float32, device=device, generator=generator).to(torch.float64)
            bf_next = (1 - t_next) * (bf_latent - t_cur * bf_v) + t_next * noise
            q_next = (1 - t_next) * (q_latent - t_cur * q_v) + t_next * noise
            tf = error_stats(tf_q_v, bf_v)
            rollout = error_stats(q_v, rollout_bf_v)
            latent = error_stats(q_next, bf_next)
            rows.append({"step": step, "t_current": float(t_cur), "t_next": float(t_next),
                "teacher_forced_v_mse": tf["mse"],
                "teacher_forced_v_reference_energy": tf["reference_energy"],
                "teacher_forced_v_nmse": tf["nmse"],
                "rollout_v_mse": rollout["mse"],
                "rollout_v_reference_energy": rollout["reference_energy"],
                "rollout_v_nmse": rollout["nmse"],
                "rollout_latent_mse": latent["mse"],
                "rollout_latent_reference_energy": latent["reference_energy"],
                "rollout_latent_nmse": latent["nmse"]})
            print(rows[-1], flush=True)
            bf_latent, q_latent = bf_next, q_next
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix('.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    OUT.with_suffix('.json').write_text(json.dumps({"prompt": PROMPT, "seed": 308, "height": 480,
        "width": 832, "frames": 81, "steps": 4, "joint_state": str(STATE), "metrics": rows}, indent=2) + "\n")
    restore_activation_processors(originals)
    print("saved", OUT)

if __name__ == '__main__':
    main()
