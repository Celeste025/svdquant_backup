#!/usr/bin/env python3
"""Held-out audit of per-timestep channel-diagonal rCM-Wan output calibration.

For cached BF16 DiT calls, fit two no-bias diagonal regressions separately at
each rCM step: q ~= w * bf16 (the paper's stated direction), and
bf16 ~= a * q (the deployable correction direction).  All sufficient
statistics are accumulated online; no DiT output tensor is written to disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from diffusers import WanPipeline, WanTransformer3DModel

MODEL = Path("/data1/models/svdquant-wjq/models/Wan2.1-T2V-1.3B-Diffusers")
RCM_TRANSFORMER = Path("/data1/models/svdquant-wjq/models/rcm-Wan2.1-T2V-1.3B-Diffusers/transformer")
CKPT = Path("/data1/models/svdquant-wjq/ckpts/rcm-wan2.1-1.3b-real-nvfp4-s16")
CACHE = Path("/data1/models/svdquant-wjq/datasets/torch.bfloat16/rcm-wan2.1-1.3b/rcm4-sigma80-g0-f77/vbench/s16/caches")
EPS = 1e-30
def as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value): return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]): return value[0]
    raise TypeError(f"Expected Tensor output, got {type(value)}")
from infer_rcm_wan_4step import load_quantized_transformer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/rcm_output_channel_scale_audit"


def nmse(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, int]:
    diff = got.float() - ref.float()
    return float(diff.square().sum()), float(ref.float().square().sum()), float(diff.square().mean()), ref.numel()


def output(model: torch.nn.Module, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    kwargs = payload["input_kwargs"]
    return as_tensor(model(
        hidden_states=payload["input_args"][0].to(device=device, dtype=torch.bfloat16),
        timestep=kwargs["timestep"].to(device=device, dtype=torch.bfloat16),
        encoder_hidden_states=kwargs["encoder_hidden_states"].to(device=device, dtype=torch.bfloat16),
        return_dict=False,
    ))


def items(cache_dir: Path, seed: int, num_prompts: int) -> tuple[list[str], dict[str, list[tuple[int, dict[str, Any]]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(cache_dir.glob("*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        label = str(obj.get("filename", path.stem)).split("-")[0]
        grouped[label].append((int(obj.get("step", -1)), obj))
    complete = {name: sorted(rows) for name, rows in grouped.items()
                if [step for step, _ in sorted(rows)] == [0, 1, 2, 3]}
    names = sorted(complete)
    random.Random(seed).shuffle(names)
    if len(names) < num_prompts or num_prompts % 2:
        raise RuntimeError(f"need an even {num_prompts} complete prompts, only {len(names)} available")
    chosen = names[:num_prompts]
    return chosen, {name: complete[name] for name in chosen}


def channel_sums(q: torch.Tensor, bf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The DiT prediction is [B,C,F,H,W]; retain exactly its C output channels.
    axes = (0, *range(2, q.ndim))
    q, bf = q.float(), bf.float()
    return (q.square().sum(axes).cpu(), bf.square().sum(axes).cpu(), (q * bf).sum(axes).cpu())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--output-dir", type=Path, default=OUT / "quick_4prompt")
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_ids, grouped = items(args.cache_dir, args.seed, args.num_prompts)
    train_ids, test_ids = prompt_ids[:len(prompt_ids)//2], prompt_ids[len(prompt_ids)//2:]
    device = torch.device("cuda")

    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.transformer = WanTransformer3DModel.from_pretrained(RCM_TRANSFORMER, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.text_encoder.to("cpu"); pipe.vae.to("cpu")
    load_quantized_transformer(pipe, args.checkpoint, MODEL)
    model = pipe.transformer.eval()

    # Fit from only the training prompts.  q2/bf2/cross are [step][channel].
    q2: dict[int, torch.Tensor] = {}; bf2: dict[int, torch.Tensor] = {}; cross: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for pid in train_ids:
            for step, payload in grouped[pid]:
                q = output(model, payload, device)
                bf = as_tensor(payload["outputs"]).to(device=device, dtype=torch.bfloat16)
                x, y, z = channel_sums(q, bf)
                q2[step] = q2.get(step, torch.zeros_like(x)) + x
                bf2[step] = bf2.get(step, torch.zeros_like(y)) + y
                cross[step] = cross.get(step, torch.zeros_like(z)) + z
                print(f"fit prompt={pid} step={step}", flush=True)
    # Fit true quantization error e = bf - q from quantized output q.
    qe = {step: cross[step] - q2[step] for step in range(4)}
    e2 = {step: (bf2[step] + q2[step] - 2 * cross[step]).clamp_min(EPS) for step in range(4)}
    a = {step: qe[step] / q2[step].clamp_min(EPS) for step in range(4)}

    rows: list[dict[str, object]] = []
    aggregate: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0., 0., 0.])
    scatter: dict[str, list[torch.Tensor]] = defaultdict(list)
    token_scatter: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for split, ids in (("train", train_ids), ("test", test_ids)):
            for pid in ids:
                for step, payload in grouped[pid]:
                    q = output(model, payload, device).float()
                    bf = as_tensor(payload["outputs"]).to(device=device).float()
                    scale_a = a[step].to(device).view(1, -1, *([1] * (q.ndim - 2)))
                    error = bf - q
                    predicted_error = scale_a * q
                    flat_error, flat_pred = error.flatten(), predicted_error.flatten()
                    sample_idx = torch.linspace(0, flat_error.numel() - 1, steps=min(2048, flat_error.numel()), device=device).long()
                    if split == "test":
                        mid_t = q.shape[2] // 2
                        for channel in range(q.shape[1]):
                            pair = torch.stack((q[0, channel, mid_t].flatten(), error[0, channel, mid_t].flatten()), dim=1).cpu()
                            token_scatter[f"step{step}_channel{channel}"].append(pair)
                    scatter[f"{split}_step{step}"].append(torch.stack((flat_error[sample_idx], flat_pred[sample_idx]), dim=1).cpu())
                    for name, got, ref in (("raw_q_vs_bf", q, bf), ("predicted_error_vs_true_error", predicted_error, error),
                                           ("calibrated_q_plus_predicted_error_vs_bf", q + predicted_error, bf)):
                        err2, ref2, mse, count = nmse(got, ref)
                        rows.append({"split": split, "prompt_id": pid, "step": step, "metric": name,
                                     "nmse": err2 / max(ref2, EPS), "mse": mse, "err2": err2, "ref2": ref2,
                                     "numel": count})
                        acc = aggregate[(split, step, name)]; acc[0] += err2; acc[1] += ref2; acc[2] += count
                    print(f"eval split={split} prompt={pid} step={step}", flush=True)

    summary = []
    for (split, step, name), (err2, ref2, count) in sorted(aggregate.items()):
        summary.append({"split": split, "step": step, "metric": name, "nmse": err2 / max(ref2, EPS),
                        "mse": err2 / count, "err2": err2, "ref2": ref2, "numel": int(count)})
    channels = []
    for step in range(4):
        residual = (e2[step] - qe[step].square() / q2[step].clamp_min(EPS)).clamp_min(0) / e2[step].clamp_min(EPS)
        for channel in range(a[step].numel()):
            channels.append({"step": step, "channel": channel, "a_q_to_error": float(a[step][channel]),
                             "error_unexplained_fraction": float(residual[channel])})
    torch.save({key: torch.cat(value, dim=0) for key, value in token_scatter.items()}, args.output_dir / "token_scatter_samples.pt")
    torch.save({key: torch.cat(value, dim=0) for key, value in scatter.items()}, args.output_dir / "scatter_samples.pt")

    with (args.output_dir / "per_prompt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (args.output_dir / "channel_scales.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(channels[0])); writer.writeheader(); writer.writerows(channels)
    (args.output_dir / "config.json").write_text(json.dumps({"prompt_ids": prompt_ids, "train_ids": train_ids,
        "test_ids": test_ids, "equations": {"error_target": "e = bf16 - q", "predictor": "e_hat = a_t * q", "calibration": "q + e_hat"},
        "checkpoint": str(args.checkpoint), "cache_dir": str(args.cache_dir)}, indent=2) + "\n")

    print(json.dumps({"train_ids": train_ids, "test_ids": test_ids, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
