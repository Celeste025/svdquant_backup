#!/usr/bin/env python3
"""Capture Wan gate vectors at selected (timestep, block) and plot heatmaps on one figure."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
os.environ.setdefault("PYTHONMALLOC", "malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DEEPCOMPRESSOR_WAN_FLOW_SHIFT", "3.0")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/ssd/2/wenjinqi.wjq"))
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_ROOT / "hf" / "hub"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

WAN_PATH = os.environ.get("WAN_MODEL_PATH", "/ssd/2/yuzhibo.yzh_data/Wan2.1-T2V-1.3B-Diffusers")
OUT_DIR = Path(os.environ.get("GATE_OUT_DIR", str(DATA_ROOT / "compare" / "wan_gate_stats")))
PROMPT = "an airplane soaring through a clear blue sky"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)

# Capture these blocks every step; later pick a few (t, block) pairs for the figure.
BLOCK_IDS = (0, 7, 14, 21, 29)
NUM_STEPS = 10
# Heatmap reshape for C=1536 -> 32x48
HEAT_H, HEAT_W = 32, 48
# Prefer conditional branch under CFG (batch: [uncond, cond]).
CFG_IDX = 1


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} out={OUT_DIR}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)
    transformer = pipe.transformer

    # records[block][kind] -> list over transformer forwards (CFG doubles each step)
    records: dict[int, dict[str, list[np.ndarray]]] = {i: {"gate_msa": [], "c_gate_msa": []} for i in BLOCK_IDS}
    originals = {}
    call_count = {"n": 0}

    def make_wrapped(idx: int, block):
        orig = block.forward
        originals[idx] = orig

        def forward(hidden_states, encoder_hidden_states, temb, rotary_emb):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                block.scale_shift_table + temb.float()
            ).chunk(6, dim=1)
            # (B,1,C) -> keep channel vector for chosen CFG branch
            g = gate_msa.detach().float().cpu().numpy()
            cg = c_gate_msa.detach().float().cpu().numpy()
            b_idx = min(CFG_IDX, g.shape[0] - 1)
            records[idx]["gate_msa"].append(g[b_idx, 0].copy())  # (C,)
            records[idx]["c_gate_msa"].append(cg[b_idx, 0].copy())
            if idx == BLOCK_IDS[0]:
                call_count["n"] += 1
            return orig(hidden_states, encoder_hidden_states, temb, rotary_emb)

        return forward

    for i in BLOCK_IDS:
        transformer.blocks[i].forward = make_wrapped(i, transformer.blocks[i])

    print(f"running {NUM_STEPS} steps, prompt={PROMPT!r}", flush=True)
    gen = torch.Generator(device=device).manual_seed(42)
    with torch.inference_mode():
        _ = pipe(
            PROMPT,
            negative_prompt=NEGATIVE,
            height=480,
            width=832,
            num_frames=33,
            num_inference_steps=NUM_STEPS,
            guidance_scale=6.0,
            generator=gen,
        )

    for i, fn in originals.items():
        transformer.blocks[i].forward = fn

    n_fwd = call_count["n"]
    # With CFG, expect 2 forwards per step.
    fwds_per_step = max(1, n_fwd // NUM_STEPS)
    print(f"forwards={n_fwd}, fwds_per_step≈{fwds_per_step}", flush=True)

    def fwd_for_step(step: int) -> int:
        # Use the conditional forward within that step: last of the CFG pair.
        return step * fwds_per_step + (fwds_per_step - 1)

    # 7 (timestep, block) pairs spanning early/mid/late × shallow/mid/deep
    pairs = [
        (0, 0),
        (0, 14),
        (0, 29),
        (4, 14),
        (9, 0),
        (9, 14),
        (9, 29),
    ]

    # Collect arrays; color scale uses robust percentile so rare tails don't wash out structure.
    mats: list[tuple[int, int, str, np.ndarray]] = []
    flat_abs = []
    for step, blk in pairs:
        fi = fwd_for_step(step)
        for kind in ("gate_msa", "c_gate_msa"):
            vec = records[blk][kind][fi]
            assert vec.size == HEAT_H * HEAT_W, f"dim={vec.size}, expected {HEAT_H * HEAT_W}"
            mats.append((step, blk, kind, vec.reshape(HEAT_H, HEAT_W)))
            flat_abs.append(np.abs(vec))
    all_abs = np.concatenate(flat_abs)
    vmax_full = float(all_abs.max())
    vmax_p995 = float(np.percentile(all_abs, 99.5))
    vmax_p995 = max(vmax_p995, 1e-3)
    vmax_full = max(vmax_full, 1e-3)

    def plot_grid(vmax: float, tag: str, color_note: str) -> Path:
        fig, axes = plt.subplots(len(pairs), 2, figsize=(11, 1.55 * len(pairs)), constrained_layout=True)
        im = None
        for row, (step, blk) in enumerate(pairs):
            for col, kind in enumerate(("gate_msa", "c_gate_msa")):
                ax = axes[row, col]
                fi = fwd_for_step(step)
                vec = records[blk][kind][fi]
                mat = vec.reshape(HEAT_H, HEAT_W)
                im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"t={step} block={blk} {kind}  "
                    f"[min={vec.min():.2f}, max={vec.max():.2f}, |p95|={np.percentile(np.abs(vec), 95):.2f}]",
                    fontsize=8,
                )
                if col == 0:
                    ax.set_ylabel(f"t{step}/b{blk}", fontsize=9)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label=f"gate (±{vmax:.2f})")
        fig.suptitle(
            f"Wan-1.3B gate vectors as {HEAT_H}×{HEAT_W} heatmaps "
            f"(CFG cond, seed=42, {NUM_STEPS} steps; {color_note})\n{PROMPT}",
            fontsize=11,
        )
        out = OUT_DIR / f"gate_vector_heatmaps_{tag}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return out

    def plot_strips(vmax: float, tag: str, color_note: str) -> Path:
        fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 1.2 * len(pairs)), constrained_layout=True)
        im = None
        for row, (step, blk) in enumerate(pairs):
            for col, kind in enumerate(("gate_msa", "c_gate_msa")):
                ax = axes[row, col]
                fi = fwd_for_step(step)
                vec = records[blk][kind][fi]
                im = ax.imshow(
                    vec[np.newaxis, :],
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.set_yticks([])
                ax.set_xlabel("channel" if row == len(pairs) - 1 else "")
                ax.set_title(f"t={step} block={blk} {kind}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"t{step}/b{blk}", fontsize=9)
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label=f"gate (±{vmax:.2f})")
        fig.suptitle(
            f"Wan-1.3B gate vectors (1×C strips, CFG cond; {color_note})\n{PROMPT}",
            fontsize=11,
        )
        out = OUT_DIR / f"gate_vector_strips_{tag}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return out

    out_clip = plot_grid(vmax_p995, "p995", "color=p99.5 clip")
    out_full = plot_grid(vmax_full, "full", "color=full range, no clip")
    out_strip_clip = plot_strips(vmax_p995, "p995", "color=p99.5 clip")
    out_strip_full = plot_strips(vmax_full, "full", "color=full range, no clip")
    # Keep previous filenames as aliases to the clipped versions for convenience.
    import shutil

    shutil.copyfile(out_clip, OUT_DIR / "gate_vector_heatmaps.png")
    shutil.copyfile(out_strip_clip, OUT_DIR / "gate_vector_strips.png")

    meta = {
        "prompt": PROMPT,
        "steps": NUM_STEPS,
        "pairs": [{"timestep": s, "block": b} for s, b in pairs],
        "cfg_branch_index": CFG_IDX,
        "reshape": [HEAT_H, HEAT_W],
        "vmax_display_p99_5": vmax_p995,
        "vmax_full": vmax_full,
        "fwds_per_step": fwds_per_step,
        "out_heatmaps_p995": str(out_clip),
        "out_heatmaps_full": str(out_full),
        "out_strips_p995": str(out_strip_clip),
        "out_strips_full": str(out_strip_full),
    }
    (OUT_DIR / "gate_vector_heatmaps_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    for p in (out_clip, out_full, out_strip_clip, out_strip_full):
        print(f"wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
