#!/usr/bin/env python3
"""Wan + Flux: per-denoising-step latent NMSE of W4A4 vs BF16 (full trajectory).

Produces a single overlay line chart: step 0 .. T-1.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DEEPCOMPRESSOR_TRANSFORMER_ONLY", "0")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_FLOW_SHIFT", "3.0")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pyarrow  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_firststep_block_nmse import (  # noqa: E402
    FLUX_PROMPT,
    WAN_NEGATIVE,
    WAN_PROMPT,
    load_flux_bf16,
    load_flux_quant,
    load_wan_bf16,
    load_wan_quant,
)

SEED = 42
WAN_CKPT = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16"
FLUX_CKPT = DATA_ROOT / "ckpts" / "flux.1-dev-int4-s16"
OUT = DATA_ROOT / "compare" / "wan_flux_multistep_latent_nmse"


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).pow(2).mean() / a.pow(2).mean().clamp_min(1e-12))


@torch.inference_mode()
def run_latents_wan(pipe, prompt: str, seed: int, steps: int = 50) -> list[torch.Tensor]:
    latents: list[torch.Tensor] = []

    def cb(_pipe, index, _timestep, kwargs):
        lat = kwargs["latents"]
        latents.append(lat.detach().to("cpu", torch.bfloat16).contiguous().clone())
        return kwargs

    pipe(
        prompt,
        negative_prompt=WAN_NEGATIVE,
        height=480,
        width=832,
        num_frames=33,
        num_inference_steps=steps,
        guidance_scale=6.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="latent",
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    assert len(latents) == steps, f"wan expected {steps} steps, got {len(latents)}"
    return latents


@torch.inference_mode()
def run_latents_flux(pipe, prompt: str, seed: int, steps: int = 50) -> list[torch.Tensor]:
    latents: list[torch.Tensor] = []

    def cb(_pipe, index, _timestep, kwargs):
        lat = kwargs["latents"]
        latents.append(lat.detach().to("cpu", torch.bfloat16).contiguous().clone())
        return kwargs

    pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=steps,
        guidance_scale=3.5,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="latent",
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    assert len(latents) == steps, f"flux expected {steps} steps, got {len(latents)}"
    return latents


def curve_vs_ref(ref: list[torch.Tensor], hyp: list[torch.Tensor]) -> list[float]:
    assert len(ref) == len(hyp)
    return [nmse(ref[i], hyp[i]) for i in range(len(ref))]


def plot_overlay(curves: dict[str, list[float]], out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    style = {
        "wan": {"color": "#1f4e79", "lw": 2.0, "marker": "o", "ms": 3.5},
        "flux": {"color": "#b85c38", "lw": 2.0, "marker": "s", "ms": 3.5},
    }
    for name, ys in curves.items():
        st = style.get(name, {"color": "#333", "lw": 1.6})
        xs = np.arange(len(ys))
        ax.plot(
            xs,
            [100.0 * y for y in ys],
            label=name,
            color=st["color"],
            linewidth=st["lw"],
            marker=st.get("marker"),
            markersize=st.get("ms", 3),
            markevery=max(1, len(ys) // 16),
        )
    ax.set_xlabel("denoising step")
    ax.set_ylabel("latent NMSE vs BF16 (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", choices=["wan", "flux", "both"], default="both")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    curves: dict[str, list[float]] = {}
    meta: dict = {"seed": args.seed, "models": {}}

    if args.model in ("wan", "both"):
        print("=== WAN ungated s16 vs BF16 (50 steps) ===", flush=True)
        bf16 = load_wan_bf16()
        print("[wan] BF16 generate...", flush=True)
        ref = run_latents_wan(bf16, WAN_PROMPT, args.seed, 50)
        del bf16
        gc.collect()
        torch.cuda.empty_cache()

        quant = load_wan_quant(WAN_CKPT)
        print("[wan] W4A4 generate...", flush=True)
        hyp = run_latents_wan(quant, WAN_PROMPT, args.seed, 50)
        del quant
        gc.collect()
        torch.cuda.empty_cache()

        ys = curve_vs_ref(ref, hyp)
        curves["wan"] = ys
        meta["models"]["wan"] = {
            "ckpt": str(WAN_CKPT),
            "prompt": WAN_PROMPT,
            "steps": 50,
            "guidance": 6.0,
            "lat_nmse": ys,
            "final_pct": 100.0 * ys[-1],
            "mean_pct": 100.0 * float(np.mean(ys)),
            "max_pct": 100.0 * float(np.max(ys)),
            "argmax_step": int(np.argmax(ys)),
        }
        print(
            f"[wan] final={meta['models']['wan']['final_pct']:.3f}% "
            f"mean={meta['models']['wan']['mean_pct']:.3f}% "
            f"max@{meta['models']['wan']['argmax_step']}={meta['models']['wan']['max_pct']:.3f}%",
            flush=True,
        )
        (out / "wan_lat_nmse.json").write_text(json.dumps(meta["models"]["wan"], indent=2))

    if args.model in ("flux", "both"):
        print("=== FLUX s16 vs BF16 (50 steps) ===", flush=True)
        bf16 = load_flux_bf16()
        print("[flux] BF16 generate...", flush=True)
        ref = run_latents_flux(bf16, FLUX_PROMPT, args.seed, 50)
        del bf16
        gc.collect()
        torch.cuda.empty_cache()

        quant = load_flux_quant(FLUX_CKPT)
        print("[flux] W4A4 generate...", flush=True)
        hyp = run_latents_flux(quant, FLUX_PROMPT, args.seed, 50)
        del quant
        gc.collect()
        torch.cuda.empty_cache()

        ys = curve_vs_ref(ref, hyp)
        curves["flux"] = ys
        meta["models"]["flux"] = {
            "ckpt": str(FLUX_CKPT),
            "prompt": FLUX_PROMPT,
            "steps": 50,
            "guidance": 3.5,
            "lat_nmse": ys,
            "final_pct": 100.0 * ys[-1],
            "mean_pct": 100.0 * float(np.mean(ys)),
            "max_pct": 100.0 * float(np.max(ys)),
            "argmax_step": int(np.argmax(ys)),
        }
        print(
            f"[flux] final={meta['models']['flux']['final_pct']:.3f}% "
            f"mean={meta['models']['flux']['mean_pct']:.3f}% "
            f"max@{meta['models']['flux']['argmax_step']}={meta['models']['flux']['max_pct']:.3f}%",
            flush=True,
        )
        (out / "flux_lat_nmse.json").write_text(json.dumps(meta["models"]["flux"], indent=2))

    if len(curves) >= 1:
        plot_overlay(
            curves,
            out / "wan_flux_latent_nmse_vs_timestep.png",
            "W4A4 vs BF16 latent NMSE over denoising steps",
        )
        # also separate panels if both
        if len(curves) == 2:
            fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), dpi=160, sharex=True)
            for ax, (name, ys), color in zip(
                axes,
                curves.items(),
                ("#1f4e79", "#b85c38"),
            ):
                ax.plot(np.arange(len(ys)), [100.0 * y for y in ys], color=color, lw=2.0, marker="o", ms=3, markevery=max(1, len(ys) // 16))
                ax.set_ylabel("latent NMSE vs BF16 (%)")
                m = meta["models"][name]
                ax.set_title(f"{name}: final={m['final_pct']:.2f}%  mean={m['mean_pct']:.2f}%  max@{m['argmax_step']}={m['max_pct']:.2f}%")
                ax.grid(True, alpha=0.35)
            axes[-1].set_xlabel("denoising step")
            fig.suptitle("Quantized model output error vs BF16 across timesteps", fontsize=12)
            fig.tight_layout()
            p2 = out / "wan_flux_latent_nmse_vs_timestep_panels.png"
            fig.savefig(p2)
            plt.close(fig)
            print(f"[plot] {p2}", flush=True)

    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
