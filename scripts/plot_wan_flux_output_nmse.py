#!/usr/bin/env python3
"""Wan + Flux: per-step model *output* NMSE of W4A4 vs BF16.

Wan (true CFG):
  - pre-CFG  = noise_cond
  - post-CFG = uncond + g * (cond - uncond)

Flux (guidance embeds, no dual-forward CFG by default):
  - output   = transformer noise_pred

Also records post-scheduler latent NMSE for reference.
Own trajectories (each model evolves its own latents), same seed/prompt as
plot_wan_flux_multistep_latent_nmse.py.
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
OUT = DATA_ROOT / "compare" / "wan_flux_output_nmse"


def nmse(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    a, b = ref.float(), hyp.float()
    return float((a - b).pow(2).mean() / a.pow(2).mean().clamp_min(1e-12))


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu", torch.bfloat16).contiguous().clone()


@torch.inference_mode()
def run_wan_traj(pipe, prompt: str, seed: int, steps: int = 50, guidance: float = 6.0):
    """Manual Wan denoise; capture pre-CFG (cond), post-CFG, and latents."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=WAN_NEGATIVE,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(dtype)

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        480,
        832,
        33,
        torch.float32,
        device,
        torch.Generator(device="cuda").manual_seed(seed),
        None,
    )

    pre_cfg: list[torch.Tensor] = []
    post_cfg: list[torch.Tensor] = []
    lats: list[torch.Tensor] = []

    for t in timesteps:
        latent_model_input = latents.to(dtype)
        timestep = t.expand(latents.shape[0])

        noise_cond = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        noise_uncond = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=negative_prompt_embeds,
            return_dict=False,
        )[0]
        noise_cfg = noise_uncond + guidance * (noise_cond - noise_uncond)

        pre_cfg.append(_to_cpu(noise_cond))
        post_cfg.append(_to_cpu(noise_cfg))

        latents = pipe.scheduler.step(noise_cfg, t, latents, return_dict=False)[0]
        lats.append(_to_cpu(latents))

    return {"pre_cfg": pre_cfg, "post_cfg": post_cfg, "latents": lats}


@torch.inference_mode()
def run_flux_traj(pipe, prompt: str, seed: int, steps: int = 50, guidance: float = 3.5):
    """Manual Flux denoise; capture transformer output + latents."""
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

    device = pipe._execution_device

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    height = width = 1024
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        torch.Generator(device="cuda").manual_seed(seed),
        None,
    )

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )

    if pipe.transformer.config.guidance_embeds:
        guidance_emb = torch.full([1], guidance, device=device, dtype=torch.float32)
        guidance_emb = guidance_emb.expand(latents.shape[0])
    else:
        guidance_emb = None

    outs: list[torch.Tensor] = []
    lats: list[torch.Tensor] = []

    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance_emb,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        outs.append(_to_cpu(noise_pred))
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        lats.append(_to_cpu(latents))

    return {"output": outs, "latents": lats}


def curve(ref: list[torch.Tensor], hyp: list[torch.Tensor]) -> list[float]:
    assert len(ref) == len(hyp)
    return [nmse(ref[i], hyp[i]) for i in range(len(ref))]


def plot_all(curves: dict[str, list[float]], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=160)
    style = {
        "wan_pre_cfg": {"color": "#5b8db8", "lw": 2.0, "ls": "--", "marker": "o", "ms": 3.0, "label": "Wan pre-CFG (cond)"},
        "wan_post_cfg": {"color": "#1f4e79", "lw": 2.2, "ls": "-", "marker": "o", "ms": 3.5, "label": "Wan post-CFG"},
        "wan_latent": {"color": "#1f4e79", "lw": 1.4, "ls": ":", "marker": None, "ms": 0, "label": "Wan latent (post-sched)"},
        "flux_output": {"color": "#b85c38", "lw": 2.2, "ls": "-", "marker": "s", "ms": 3.5, "label": "Flux output"},
        "flux_latent": {"color": "#b85c38", "lw": 1.4, "ls": ":", "marker": None, "ms": 0, "label": "Flux latent (post-sched)"},
    }
    for name, ys in curves.items():
        st = style.get(name, {"color": "#333", "lw": 1.6, "ls": "-", "label": name})
        xs = np.arange(len(ys))
        ax.plot(
            xs,
            [100.0 * y for y in ys],
            color=st["color"],
            lw=st["lw"],
            ls=st.get("ls", "-"),
            marker=st.get("marker"),
            markersize=st.get("ms", 0),
            label=st.get("label", name),
        )
        print(f"  {name}: first={ys[0]*100:.4f}%  mid={ys[len(ys)//2]*100:.4f}%  last={ys[-1]*100:.4f}%")

    ax.set_xlabel("denoising step")
    ax.set_ylabel("NMSE vs BF16 (%)")
    ax.set_title("W4A4 vs BF16 — model output NMSE (Wan: pre/post CFG) + latent ref")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] {out_png}")


def free(*objs):
    for o in objs:
        try:
            if hasattr(o, "to"):
                try:
                    o.to("cpu")
                except Exception:
                    pass
            if hasattr(o, "transformer"):
                try:
                    o.transformer.to("cpu")
                except Exception:
                    pass
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--skip-wan", action="store_true")
    ap.add_argument("--skip-flux", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    OUT.mkdir(parents=True, exist_ok=True)
    steps = args.steps
    curves: dict[str, list[float]] = {}
    summary: dict = {"seed": SEED, "steps": steps, "gpu": args.gpu}

    try:
        if not args.skip_wan:
            print("[wan] load BF16 …")
            pipe_bf = load_wan_bf16()
            print("[wan] BF16 traj …")
            bf = run_wan_traj(pipe_bf, WAN_PROMPT, SEED, steps)
            free(pipe_bf)

            print("[wan] load W4A4 …")
            pipe_q = load_wan_quant(WAN_CKPT)
            print("[wan] W4A4 traj …")
            qq = run_wan_traj(pipe_q, WAN_PROMPT, SEED, steps)
            free(pipe_q)

            curves["wan_pre_cfg"] = curve(bf["pre_cfg"], qq["pre_cfg"])
            curves["wan_post_cfg"] = curve(bf["post_cfg"], qq["post_cfg"])
            curves["wan_latent"] = curve(bf["latents"], qq["latents"])
            summary["wan"] = {
                "pre_cfg": curves["wan_pre_cfg"],
                "post_cfg": curves["wan_post_cfg"],
                "latent": curves["wan_latent"],
                "prompt": WAN_PROMPT,
                "guidance": 6.0,
            }
            (OUT / "wan_partial.json").write_text(
                json.dumps({"wan": summary["wan"], "seed": SEED, "steps": steps}, indent=2)
            )
            print(f"[wan] saved {OUT / 'wan_partial.json'}", flush=True)
            free(bf, qq)

        if args.skip_wan and (OUT / "wan_partial.json").exists():
            prev = json.loads((OUT / "wan_partial.json").read_text())
            summary["wan"] = prev["wan"]
            curves["wan_pre_cfg"] = prev["wan"]["pre_cfg"]
            curves["wan_post_cfg"] = prev["wan"]["post_cfg"]
            curves["wan_latent"] = prev["wan"]["latent"]
            print("[wan] loaded wan_partial.json", flush=True)

        if not args.skip_flux:
            print("[flux] load BF16 …")
            pipe_bf = load_flux_bf16()
            print("[flux] BF16 traj …")
            bf = run_flux_traj(pipe_bf, FLUX_PROMPT, SEED, steps)
            free(pipe_bf)

            print("[flux] load W4A4 …")
            pipe_q = load_flux_quant(FLUX_CKPT)
            print("[flux] W4A4 traj …")
            qq = run_flux_traj(pipe_q, FLUX_PROMPT, SEED, steps)
            free(pipe_q)

            curves["flux_output"] = curve(bf["output"], qq["output"])
            curves["flux_latent"] = curve(bf["latents"], qq["latents"])
            summary["flux"] = {
                "output": curves["flux_output"],
                "latent": curves["flux_latent"],
                "prompt": FLUX_PROMPT,
                "guidance": 3.5,
            }
            free(bf, qq)

        out_png = OUT / "wan_flux_output_nmse_vs_timestep.png"
        plot_all(curves, out_png)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[ok] {OUT / 'summary.json'}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
