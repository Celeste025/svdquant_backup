#!/usr/bin/env python3
"""First-timestep Wan noise NMSE vs BF16, before vs after CFG.

For ungated-s16 and each keep-BF16 block group, run ONLY denoising step 0:
  noise_cond   = transformer(latents, prompt_embeds)
  noise_uncond = transformer(latents, negative_embeds)
  noise_cfg    = uncond + guidance * (cond - uncond)

Compare each to BF16 counterparts. Tests whether CFG pulls different keep-BF16
hybrids' errors closer together.
"""
from __future__ import annotations

import argparse
import copy
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
    WAN_NEGATIVE,
    WAN_PROMPT,
    load_wan_bf16,
    load_wan_quant,
)
from exp_wan_keep_bf16_groups import CONFIGS, apply_blocks, snapshot_blocks  # noqa: E402

SEED = 42
GUIDANCE = 6.0
CKPT = DATA_ROOT / "ckpts" / "wan2.1-1.3b-int4-s16"
OUT = DATA_ROOT / "compare" / "wan_firststep_cfg_nmse"


def nmse(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    a, b = ref.float(), hyp.float()
    return float((a - b).pow(2).mean() / a.pow(2).mean().clamp_min(1e-12))


def cosine(ref: torch.Tensor, hyp: torch.Tensor) -> float:
    a, b = ref.float().flatten(), hyp.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


@torch.inference_mode()
def prepare_step0_inputs(pipe, prompt: str, negative: str, seed: int):
    """Mirror WanPipeline step-0 inputs (shared across models)."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=device,
    )
    # prepare latents like pipeline
    batch_size = 1
    num_channels_latents = pipe.transformer.config.in_channels
    height, width, num_frames = 480, 832, 33
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latents = pipe.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )
    pipe.scheduler.set_timesteps(50, device=device)
    t = pipe.scheduler.timesteps[0]
    timestep = t.expand(latents.shape[0])
    latent_model_input = latents.to(dtype)
    return {
        "latents": latents,
        "latent_model_input": latent_model_input,
        "timestep": timestep,
        "prompt_embeds": prompt_embeds.to(dtype),
        "negative_prompt_embeds": negative_prompt_embeds.to(dtype),
        "t": t,
    }


@torch.inference_mode()
def predict_step0(pipe, inputs: dict, guidance: float) -> dict[str, torch.Tensor]:
    """Return CPU bf16 tensors: cond, uncond, cfg."""
    tr = pipe.transformer
    cond = tr(
        hidden_states=inputs["latent_model_input"],
        timestep=inputs["timestep"],
        encoder_hidden_states=inputs["prompt_embeds"],
        return_dict=False,
    )[0]
    uncond = tr(
        hidden_states=inputs["latent_model_input"],
        timestep=inputs["timestep"],
        encoder_hidden_states=inputs["negative_prompt_embeds"],
        return_dict=False,
    )[0]
    cfg = uncond + guidance * (cond - uncond)
    to_cpu = lambda x: x.detach().to("cpu", torch.bfloat16).contiguous().clone()
    return {"cond": to_cpu(cond), "uncond": to_cpu(uncond), "cfg": to_cpu(cfg)}


def compare_pack(ref: dict[str, torch.Tensor], hyp: dict[str, torch.Tensor]) -> dict:
    out = {}
    for k in ("cond", "uncond", "cfg"):
        out[k] = {
            "nmse": nmse(ref[k], hyp[k]),
            "nmse_pct": 100.0 * nmse(ref[k], hyp[k]),
            "cosine": cosine(ref[k], hyp[k]),
        }
    # CFG amplification of absolute error energy vs cond
    e_cond = (ref["cond"].float() - hyp["cond"].float()).pow(2).mean().item()
    e_cfg = (ref["cfg"].float() - hyp["cfg"].float()).pow(2).mean().item()
    out["mse_cfg_over_cond"] = float(e_cfg / max(e_cond, 1e-20))
    return out


def pairwise_nmse_matrix(packs: dict[str, dict[str, torch.Tensor]], key: str) -> dict:
    tags = list(packs.keys())
    mat = {}
    for a in tags:
        mat[a] = {}
        for b in tags:
            mat[a][b] = 100.0 * nmse(packs[a][key], packs[b][key])
    return mat


def plot_bars(rows: list[dict], out_png: Path) -> None:
    tags = [r["tag"] for r in rows]
    x = np.arange(len(tags))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    for i, (k, color) in enumerate(
        (("cond", "#1f4e79"), ("uncond", "#2a9d8f"), ("cfg", "#e76f51"))
    ):
        ys = [r["vs_bf16"][k]["nmse_pct"] for r in rows]
        ax.bar(x + (i - 1) * w, ys, width=w, label=f"noise_{k}", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("NMSE vs BF16 (%)")
    ax.set_title(f"Wan first-step noise NMSE vs BF16 (guidance={GUIDANCE})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def plot_pairwise(mat_cond: dict, mat_cfg: dict, out_png: Path) -> None:
    tags = list(mat_cond.keys())
    # off-diagonal mean
    def offdiag_mean(mat):
        vals = []
        for i, a in enumerate(tags):
            for j, b in enumerate(tags):
                if i < j:
                    vals.append(mat[a][b])
        return float(np.mean(vals)) if vals else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    for ax, mat, title in (
        (axes[0], mat_cond, f"pairwise NMSE on noise_cond (mean od={offdiag_mean(mat_cond):.2f}%)"),
        (axes[1], mat_cfg, f"pairwise NMSE on noise_cfg (mean od={offdiag_mean(mat_cfg):.2f}%)"),
    ):
        arr = np.array([[mat[a][b] for b in tags] for a in tags], dtype=float)
        im = ax.imshow(arr, cmap="magma")
        ax.set_xticks(range(len(tags)))
        ax.set_yticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tags, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(tags)):
            for j in range(len(tags)):
                ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Do hybrids get closer after CFG? (pairwise % NMSE)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[plot] {out_png}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] load BF16 + prepare shared step-0 inputs...", flush=True)
    bf16 = load_wan_bf16()
    inputs = prepare_step0_inputs(bf16, WAN_PROMPT, WAN_NEGATIVE, SEED)
    # freeze latent / embeds on CPU copies for reuse (same tensors)
    shared = {
        "latent_model_input": inputs["latent_model_input"].clone(),
        "timestep": inputs["timestep"].clone(),
        "prompt_embeds": inputs["prompt_embeds"].clone(),
        "negative_prompt_embeds": inputs["negative_prompt_embeds"].clone(),
    }
    print("[bf16] predict step0...", flush=True)
    ref = predict_step0(bf16, shared, GUIDANCE)
    del bf16
    gc.collect()
    torch.cuda.empty_cache()

    print("[2/3] load quant + snapshot all blocks needed...", flush=True)
    quant = load_wan_quant(CKPT)
    # restore shared inputs onto quant device/dtype
    dtype = quant.transformer.dtype
    device = next(quant.transformer.parameters()).device
    shared_q = {
        "latent_model_input": shared["latent_model_input"].to(device=device, dtype=dtype),
        "timestep": shared["timestep"].to(device=device),
        "prompt_embeds": shared["prompt_embeds"].to(device=device, dtype=dtype),
        "negative_prompt_embeds": shared["negative_prompt_embeds"].to(device=device, dtype=dtype),
    }

    # BF16 block snapshots for all groups
    all_idx = sorted({i for _, idxs in CONFIGS for i in idxs})
    # Need BF16 blocks — reload bf16 briefly for snapshots only
    print("[bf16] reload for block snapshots...", flush=True)
    bf16 = load_wan_bf16()
    snaps = snapshot_blocks(bf16.transformer, all_idx)
    del bf16
    gc.collect()
    torch.cuda.empty_cache()

    # also snapshot quant blocks to restore between configs
    quant_snaps = snapshot_blocks(quant.transformer, all_idx)

    packs: dict[str, dict[str, torch.Tensor]] = {"bf16": ref}
    rows: list[dict] = []

    configs = [("ungated", [])] + list(CONFIGS)
    for tag, keep in configs:
        print(f"\n=== [{tag}] keep={keep} ===", flush=True)
        # restore all previously touched blocks to quant, then apply keep
        apply_blocks(quant.transformer, quant_snaps, all_idx)
        if keep:
            apply_blocks(quant.transformer, snaps, keep)
        hyp = predict_step0(quant, shared_q, GUIDANCE)
        packs[tag] = hyp
        vs = compare_pack(ref, hyp)
        row = {"tag": tag, "keep_bf16_blocks": keep, "vs_bf16": vs}
        rows.append(row)
        print(
            f"  cond={vs['cond']['nmse_pct']:.3f}%  uncond={vs['uncond']['nmse_pct']:.3f}%  "
            f"cfg={vs['cfg']['nmse_pct']:.3f}%  mse_cfg/cond={vs['mse_cfg_over_cond']:.3f}x",
            flush=True,
        )

    # pairwise among hybrids only (exclude bf16)
    hybrid_packs = {k: v for k, v in packs.items() if k != "bf16"}
    mat_cond = pairwise_nmse_matrix(hybrid_packs, "cond")
    mat_cfg = pairwise_nmse_matrix(hybrid_packs, "cfg")

    def offdiag_mean(mat):
        tags = list(mat.keys())
        vals = [mat[a][b] for i, a in enumerate(tags) for j, b in enumerate(tags) if i < j]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "prompt": WAN_PROMPT,
        "seed": SEED,
        "guidance": GUIDANCE,
        "ckpt": str(CKPT),
        "rows": rows,
        "pairwise_cond_pct": mat_cond,
        "pairwise_cfg_pct": mat_cfg,
        "pairwise_offdiag_mean_cond_pct": offdiag_mean(mat_cond),
        "pairwise_offdiag_mean_cfg_pct": offdiag_mean(mat_cfg),
        "note": (
            "noise_cfg = uncond + w*(cond-uncond). If CFG pulled hybrids together, "
            "pairwise_offdiag_mean_cfg << pairwise_offdiag_mean_cond, and vs_bf16 cfg "
            "gaps across tags would shrink vs cond."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {out / 'summary.json'}", flush=True)
    print(
        f"pairwise off-diag mean: cond={summary['pairwise_offdiag_mean_cond_pct']:.3f}%  "
        f"cfg={summary['pairwise_offdiag_mean_cfg_pct']:.3f}%",
        flush=True,
    )

    plot_bars(rows, out / "firststep_noise_nmse_pre_post_cfg.png")
    plot_pairwise(mat_cond, mat_cfg, out / "firststep_hybrid_pairwise_pre_post_cfg.png")

    # compact TSV
    lines = ["tag\tcond%\tuncond%\tcfg%\tmse_cfg/cond"]
    for r in rows:
        v = r["vs_bf16"]
        lines.append(
            f"{r['tag']}\t{v['cond']['nmse_pct']:.4f}\t{v['uncond']['nmse_pct']:.4f}\t"
            f"{v['cfg']['nmse_pct']:.4f}\t{v['mse_cfg_over_cond']:.4f}"
        )
    (out / "summary.tsv").write_text("\n".join(lines) + "\n")
    print((out / "summary.tsv").read_text())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
