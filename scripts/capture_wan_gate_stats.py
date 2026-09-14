#!/usr/bin/env python3
"""Capture Wan block gate_msa / c_gate_msa during a short denoising run and plot distributions."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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
# Capture a few representative blocks + all steps (short run).
BLOCK_IDS = (0, 7, 14, 21, 29)
NUM_STEPS = 10  # enough to see early/mid timestep mix; full 50 not needed for gate stats


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} out={OUT_DIR}", flush=True)

    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_PATH, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe = pipe.to(device)
    transformer = pipe.transformer

    # records[block_idx][kind] -> list of flat np arrays (one per forward)
    records: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    originals = {}

    def make_wrapped(idx: int, block):
        orig = block.forward
        originals[idx] = orig

        def forward(hidden_states, encoder_hidden_states, temb, rotary_emb):
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                block.scale_shift_table + temb.float()
            ).chunk(6, dim=1)
            # Store float32 CPU copies; shape (B,1,C) -> flatten channels (and batch if >1)
            records[idx]["gate_msa"].append(gate_msa.detach().float().cpu().numpy().reshape(-1))
            records[idx]["c_gate_msa"].append(c_gate_msa.detach().float().cpu().numpy().reshape(-1))
            records[idx]["scale_msa"].append(scale_msa.detach().float().cpu().numpy().reshape(-1))
            records[idx]["c_scale_msa"].append(c_scale_msa.detach().float().cpu().numpy().reshape(-1))
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

    # Restore forwards
    for i, fn in originals.items():
        transformer.blocks[i].forward = fn

    # Summarize
    summary = {"prompt": PROMPT, "steps": NUM_STEPS, "blocks": {}, "note": "values are raw gate tensors (B,1,C) flattened"}
    for i in BLOCK_IDS:
        for kind in ("gate_msa", "c_gate_msa"):
            arr = np.concatenate(records[i][kind], axis=0)
            stats = {
                "n": int(arr.size),
                "n_calls": len(records[i][kind]),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "p01": float(np.percentile(arr, 1)),
                "p05": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "frac_abs_gt_1": float(np.mean(np.abs(arr) > 1.0)),
                "frac_abs_gt_2": float(np.mean(np.abs(arr) > 2.0)),
                "frac_abs_gt_5": float(np.mean(np.abs(arr) > 5.0)),
            }
            summary["blocks"].setdefault(str(i), {})[kind] = stats
            np.save(OUT_DIR / f"block{i:02d}_{kind}.npy", arr)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: histograms per block for gate_msa and c_gate_msa
    fig, axes = plt.subplots(len(BLOCK_IDS), 2, figsize=(10, 2.4 * len(BLOCK_IDS)), sharex=False)
    for row, i in enumerate(BLOCK_IDS):
        for col, kind in enumerate(("gate_msa", "c_gate_msa")):
            ax = axes[row, col]
            arr = np.concatenate(records[i][kind], axis=0)
            # clip view for readability but annotate full range
            lo, hi = np.percentile(arr, [0.5, 99.5])
            ax.hist(arr, bins=80, range=(float(lo), float(hi)), color="#3b6ea5" if col == 0 else "#c45c26", alpha=0.85)
            ax.axvline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.axvline(-1.0, color="k", ls="--", lw=0.8, alpha=0.6)
            st = summary["blocks"][str(i)][kind]
            ax.set_title(
                f"block{i} {kind} | mean={st['mean']:.3f} std={st['std']:.3f} "
                f"|p95|~{abs(st['p95']):.2f} max={st['max']:.2f} |g|>1: {st['frac_abs_gt_1']:.1%}",
                fontsize=9,
            )
            if row == len(BLOCK_IDS) - 1:
                ax.set_xlabel("gate value")
            ax.set_ylabel("count")
    fig.suptitle(f"Wan-1.3B gates over {NUM_STEPS} steps (seed=42)\n{PROMPT}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_hist_by_block.png", dpi=140)
    plt.close(fig)

    # Figure 2: |gate| percentiles across blocks
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(BLOCK_IDS))
    width = 0.35
    for offset, kind, color in ((-width / 2, "gate_msa", "#3b6ea5"), (width / 2, "c_gate_msa", "#c45c26")):
        p50 = [abs(summary["blocks"][str(i)][kind]["p50"]) for i in BLOCK_IDS]
        p95 = [abs(summary["blocks"][str(i)][kind]["p95"]) for i in BLOCK_IDS]
        p99 = [abs(summary["blocks"][str(i)][kind]["p99"]) for i in BLOCK_IDS]
        ax.bar(x + offset, p95, width=width, label=f"|{kind}| p95", color=color, alpha=0.85)
        ax.scatter(x + offset, p50, color="white", edgecolor=color, zorder=3, s=28, label=f"|{kind}| p50" if offset < 0 else None)
        ax.scatter(x + offset, p99, color=color, marker="_", s=120, zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=0.9, alpha=0.7, label="|g|=1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"b{i}" for i in BLOCK_IDS])
    ax.set_ylabel("|gate|")
    ax.set_title("Wan gate magnitude by block (markers: p50 points, bars: p95, ticks: p99)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_abs_percentiles.png", dpi=140)
    plt.close(fig)

    # Figure 3: mean |gate| over denoising step index (for mid block)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, kind in zip(axes, ("gate_msa", "c_gate_msa")):
        for i in BLOCK_IDS:
            # each call is one transformer forward; with CFG, usually 2 forwards/step → group later if needed
            means = [float(np.mean(np.abs(a))) for a in records[i][kind]]
            ax.plot(means, label=f"b{i}", alpha=0.85)
        ax.set_title(kind)
        ax.set_xlabel("transformer forward index (includes CFG doubles)")
        ax.set_ylabel("mean |gate|")
        ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.legend(fontsize=8)
    fig.suptitle("mean |gate| across forwards")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_abs_mean_over_time.png", dpi=140)
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote plots under {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
