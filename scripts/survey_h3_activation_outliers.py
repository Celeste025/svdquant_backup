#!/usr/bin/env python3
"""Capture MiniMax-H3 DiT internal activations for outlier inspection.

Runs the bf16 pruned pipeline on a few VBench prompts and records, for a small
set of linear-layer outputs, a downsampled 2D grid plus peak statistics at every
denoising step. Saved payloads feed scripts/render_h3_activation_3d.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minimax_h3_svdquant_common import DATA_ROOT, load_h3_pipeline

CASES_DEFAULT = ["vbench_001", "vbench_076", "vbench_166"]
TARGET_BLOCKS = [0, 24, 49]
TARGET_SUFFIXES = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
GRID_ROWS = 128
GRID_COLS = 256
ROW_TOP = 32
SPIKE_TOP = 8
HIST_BINS = 48
HIST_MIN = -8.0
HIST_MAX = 8.0
HIST_FLOOR = 1e-8
METADATA = DATA_ROOT / "videos/svdquant-videoeval-minimax-h3/metadata/videos.jsonl"
OUTPUT_DIR_DEFAULT = DATA_ROOT / "runs/h3-activation-outliers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=CASES_DEFAULT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=39)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--vram-limit-gib", type=float, default=None,
                        help="device-wide VRAM ceiling for DiffSynth offload; lower it when the GPU is shared")
    parser.add_argument("--overwrite", action="store_true", help="re-capture cases that already have outputs")
    return parser.parse_args()


def load_prompts(cases: list[str]) -> dict[str, dict]:
    prompts: dict[str, dict] = {}
    for line in METADATA.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        sidecar = row.get("sidecar") or {}
        case_id = sidecar.get("case_id")
        if case_id in cases and case_id not in prompts and sidecar.get("prompt"):
            prompts[case_id] = {"prompt": sidecar["prompt"], "seed": int(sidecar.get("seed", 0))}
    missing = [case for case in cases if case not in prompts]
    if missing:
        raise KeyError(f"missing prompts for {missing}")
    return prompts


def grid_indices(size: int, target: int) -> torch.Tensor:
    if size <= target:
        return torch.arange(size)
    return torch.linspace(0, size - 1, target).round().long()


def flatten_output(output) -> torch.Tensor:
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(tensor):
        raise TypeError(f"unexpected activation type {type(tensor)!r}")
    if tensor.dim() == 3:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if tensor.dim() != 2:
        raise ValueError(f"unexpected activation shape {tuple(tensor.shape)}")
    return tensor.detach()


def capture(tensor: torch.Tensor) -> dict:
    rows = grid_indices(tensor.shape[0], GRID_ROWS).to(tensor.device)
    cols = grid_indices(tensor.shape[1], GRID_COLS).to(tensor.device)
    grid = tensor[rows][:, cols].float().cpu()
    col_peak = tensor.abs().amax(dim=0).float().cpu()
    top = torch.topk(col_peak, k=min(GRID_COLS, col_peak.numel())).indices
    grid_sorted = tensor[rows][:, top.to(tensor.device)].float().cpu()
    abs_tensor = tensor.abs()
    absmax = float(abs_tensor.max())
    mean_abs = float(abs_tensor.mean())
    row_peak = abs_tensor.amax(dim=1).float().cpu()
    row_top = torch.topk(row_peak, k=min(ROW_TOP, row_peak.numel())).indices.long()
    # Channel-wise typical scale, and how far the worst tokens exceed it.  A token
    # that is huge in *many* channels cannot be fixed by channel-wise smoothing.
    col_median = abs_tensor.median(dim=0).values.float().cpu()
    spike_tokens = row_top[:SPIKE_TOP].long()
    spike_rows = abs_tensor[spike_tokens.to(tensor.device)]
    spike_ratio = (spike_rows / col_median.to(tensor.device).clamp_min(1e-12)).float()
    spike_stats = torch.stack([
        row_peak[spike_tokens],
        spike_rows.mean(dim=1).float().cpu(),
        spike_ratio.amax(dim=1).cpu(),
        (spike_ratio > 10).sum(dim=1).float().cpu(),
        (spike_ratio > 100).sum(dim=1).float().cpu(),
    ], dim=1).double()
    histogram = torch.histc(torch.log10(abs_tensor.float().clamp_min(HIST_FLOOR)), bins=HIST_BINS,
                            min=HIST_MIN, max=HIST_MAX)
    return {
        "grid": grid,
        "grid_sorted": grid_sorted,
        "col_peak": col_peak,
        "row_peak": row_peak,
        "row_top": row_top,
        "col_median": col_median,
        "spike_tokens": spike_tokens.cpu().long(),
        "spike_stats": spike_stats,
        "abs_hist": histogram.double().cpu(),
        "top_channels": top.long(),
        "stats": {
            "shape": [int(tensor.shape[0]), int(tensor.shape[1])],
            "absmax": absmax,
            "mean_abs": mean_abs,
            "peak_over_mean": absmax / (mean_abs + 1e-12),
            "grid_absmax": float(grid.abs().max()),
        },
    }


def compute_layout(pipe, prompt: str, height: int, width: int, frames: int) -> dict[str, list[int]]:
    """Boundaries of the packed sequence: [text | audio | video | pad] (fl2va, no keyframes)."""
    from diffsynth.models.minimax_h3_text_encoder import presentation_t2va

    text_len = len(presentation_t2va(pipe.tokenizer, prompt)[0])
    latent_t = ((frames - 5) // 17) * 5 + 2
    frame_rows = (height // 16 // 2) * (width // 16 // 2)
    audio_rows = round(frames / 24.0 * 40.0) * 2
    video_rows = latent_t * frame_rows
    used = text_len + audio_rows + video_rows
    seq_len = ((used + 63) // 64) * 64
    return {
        "text": [0, text_len],
        "audio": [text_len, text_len + audio_rows],
        "video": [text_len + audio_rows, used],
        "pad": [used, seq_len],
    }


def export_npz(payload: dict, path: Path) -> None:
    import numpy as np

    arrays: dict[str, np.ndarray] = {}
    for step, records in payload["steps"].items():
        for layer, record in records.items():
            prefix = f"s{step:02d}/{layer}"
            arrays[f"{prefix}/grid"] = record["grid"].numpy()
            arrays[f"{prefix}/grid_sorted"] = record["grid_sorted"].numpy()
            arrays[f"{prefix}/col_peak"] = record["col_peak"].numpy()
            arrays[f"{prefix}/row_peak"] = record["row_peak"].numpy()
            arrays[f"{prefix}/row_top"] = record["row_top"].numpy()
            arrays[f"{prefix}/col_median"] = record["col_median"].numpy()
            arrays[f"{prefix}/spike_tokens"] = record["spike_tokens"].numpy()
            arrays[f"{prefix}/spike_stats"] = record["spike_stats"].numpy()
            arrays[f"{prefix}/abs_hist"] = record["abs_hist"].numpy()
            arrays[f"{prefix}/top_channels"] = record["top_channels"].numpy()
            stats = record["stats"]
            arrays[f"{prefix}/stats"] = np.array(
                [stats["absmax"], stats["mean_abs"], stats["peak_over_mean"], stats["grid_absmax"], *stats["shape"]],
                dtype=np.float64,
            )
    np.savez_compressed(path, case=payload["case"], prompt=payload["prompt"], targets=np.array(payload["targets"]),
                        layout=np.array(json.dumps(payload["layout"])), **arrays)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        case for case in args.cases
        if args.overwrite
        or not ((args.output_dir / f"{case}.pt").is_file() and (args.output_dir / f"{case}.npz").is_file())
    ]
    if not pending:
        print(f"all cases already captured in {args.output_dir}; nothing to do", flush=True)
        return
    if len(pending) != len(args.cases):
        print(f"skipping captured cases: {sorted(set(args.cases) - set(pending))}", flush=True)

    pipe = load_h3_pipeline(full=True, vram_limit_gib=args.vram_limit_gib)
    print("pipeline loaded", flush=True)

    layers: list[tuple[str, torch.nn.Module]] = []
    for block_id in TARGET_BLOCKS:
        block = pipe.dit.blocks[block_id]
        for suffix in TARGET_SUFFIXES:
            module = block
            for part in suffix.split("."):
                module = getattr(module, part)
            layers.append((f"blocks.{block_id}.{suffix}", module))
    print(f"hooked layers: {[name for name, _ in layers]}", flush=True)

    for case in pending:
        row = prompts[case]
        steps: dict[int, dict[str, dict]] = {}
        state = {"step": -1}

        def layer_hook(layer_name: str):
            def hook(_module, _inputs, output):
                step = state["step"]
                try:
                    tensor = flatten_output(output)
                except Exception as error:
                    raise RuntimeError(f"{layer_name} step {step}: {error}") from error
                steps.setdefault(step, {})[layer_name] = capture(tensor)
                if step == 0 and layer_name == layers[0][0]:
                    print(f"  step0 {layer_name} shape={tuple(tensor.shape)} absmax={float(tensor.abs().max()):.3f}", flush=True)
            return hook

        handles = [module.register_forward_hook(layer_hook(name)) for name, module in layers]

        def step_counter(_module, _inputs, _kwargs):
            state["step"] += 1

        counter_handle = pipe.dit.register_forward_pre_hook(step_counter, with_kwargs=True)
        try:
            pipe(
                prompt=row["prompt"], seed=row["seed"], height=args.height, width=args.width,
                num_frames=args.frames, num_inference_steps=args.steps, cfg_scale=1.0, tiled=True,
            )
        finally:
            counter_handle.remove()
            for handle in handles:
                handle.remove()

        captured = sorted(steps)
        if captured != list(range(args.steps)):
            raise RuntimeError(f"{case}: expected steps 0..{args.steps - 1}, captured {captured}")

        payload = {
            "case": case,
            "prompt": row["prompt"],
            "seed": row["seed"],
            "config": {"height": args.height, "width": args.width, "frames": args.frames, "steps": args.steps},
            "layout": compute_layout(pipe, row["prompt"], args.height, args.width, args.frames),
            "targets": [name for name, _ in layers],
            "steps": steps,
        }
        target = args.output_dir / f"{case}.pt"
        torch.save(payload, target)
        export_npz(payload, args.output_dir / f"{case}.npz")
        summary = {
            step: {layer: round(record["stats"]["absmax"], 3) for layer, record in records.items()}
            for step, records in sorted(steps.items())
        }
        print(f"[{case}] saved {target.name}; steps captured={len(steps)}; step0 absmax={summary.get(0)}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
