#!/usr/bin/env python3
"""One real H3 SVDQuant generation with per-linear dynamic NVFP4 scale audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from infer_minimax_h3_svdquant_standard import apply_svdquant
from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt, target_linears
from diffsynth.utils.data.audio_video import write_video_audio


THRESHOLD = 6.0 * 448.0


class ScaleRecorder:
    def __init__(self, raw_path: Path):
        self.raw_path = raw_path
        self.pending: list[dict] = []
        self.calls: Counter[str] = Counter()
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        if self.raw_path.exists():
            raise FileExistsError(self.raw_path)

    def observer(self, name: str):
        def record(x: torch.Tensor, absmax: torch.Tensor, scale: torch.Tensor) -> None:
            call = self.calls[name]
            self.calls[name] += 1
            self.pending.append({
                "layer": name,
                "call_index": call,
                "elements": x.numel(),
                "absmax": absmax.detach(),
                "tensor_scale": scale.detach(),
                "above_2688": torch.count_nonzero(x.abs() > THRESHOLD).detach(),
            })
            if len(self.pending) >= 200:
                self.flush()
        return record

    def flush(self) -> None:
        if not self.pending:
            return
        maxes = torch.stack([r["absmax"] for r in self.pending]).cpu().tolist()
        scales = torch.stack([r["tensor_scale"] for r in self.pending]).cpu().tolist()
        counts = torch.stack([r["above_2688"] for r in self.pending]).cpu().tolist()
        with self.raw_path.open("a", encoding="utf-8") as stream:
            for r, maxval, scale, count in zip(self.pending, maxes, scales, counts, strict=True):
                stream.write(json.dumps({
                    "layer": r["layer"], "call_index": r["call_index"],
                    "elements": r["elements"], "absmax_x_over_s": maxval,
                    "tensor_scale": scale, "above_2688": count,
                }) + "\n")
        self.pending.clear()
        print(f"audit rows saved: {sum(self.calls.values())}", flush=True)


def quantiles(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    return {f"p{p}": s[round((len(s) - 1) * p / 100)] for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def summarize(raw_path: Path) -> dict:
    rows = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
    if len(rows) != 200 * 20:
        raise RuntimeError(f"expected 4,000 linear calls, observed {len(rows)}")
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)
        if not (row["elements"] > 0 and row["absmax_x_over_s"] >= 0 and row["tensor_scale"] > 0):
            raise RuntimeError(f"invalid audit row: {row}")
    if len(by_layer) != 200 or any(sorted(r["call_index"] for r in group) != list(range(20))
                                   for group in by_layer.values()):
        raise RuntimeError("missing or duplicated layer/step calls")
    total_elements = sum(row["elements"] for row in rows)
    exceed = sum(row["above_2688"] for row in rows)
    scales = [row["tensor_scale"] for row in rows]
    maxes = [row["absmax_x_over_s"] for row in rows]
    return {
        "calls": len(rows), "layers": len(by_layer), "calls_per_layer": 20,
        "threshold_abs_x_over_s": THRESHOLD,
        "total_activation_elements": total_elements,
        "elements_above_2688": exceed,
        "fraction_elements_above_2688": exceed / total_elements,
        "calls_with_any_above_2688": sum(row["above_2688"] > 0 for row in rows),
        "max_abs_x_over_s": max(maxes),
        "absmax_x_over_s_quantiles": quantiles(maxes),
        "tensor_scale_quantiles": quantiles(scales),
        "tensor_scale_above_one_calls": sum(s > 1 for s in scales),
        "per_layer": {name: {
            "elements_above_2688": sum(r["above_2688"] for r in group),
            "calls_with_any_above_2688": sum(r["above_2688"] > 0 for r in group),
            "max_abs_x_over_s": max(r["absmax_x_over_s"] for r in group),
            "tensor_scale_quantiles": quantiles([r["tensor_scale"] for r in group]),
        } for name, group in sorted(by_layer.items())},
        "per_call_index": {str(i): {
            "elements_above_2688": sum(r["above_2688"] for r in rows if r["call_index"] == i),
            "calls_with_any_above_2688": sum(r["above_2688"] > 0 for r in rows if r["call_index"] == i),
            "tensor_scale_quantiles": quantiles([r["tensor_scale"] for r in rows if r["call_index"] == i]),
        } for i in range(20)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-id", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "activation_scale_calls.jsonl"
    report_path = args.output_dir / "summary.json"
    video_path = args.output_dir / "svdquant_480x832_124f_20steps.mp4"
    if report_path.exists() or video_path.exists():
        raise FileExistsError("output already exists; choose a fresh output directory")
    row = read_prompt(args.prompt_id)
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    if state.get("format") != "minimax-h3-svdquant-standard-v1" or len(state.get("layers", {})) != 200 \
            or config.get("rank") != 64:
        raise RuntimeError(f"expected complete rank-64 H3 SVDQuant state, got {config}")
    del state
    pipe = load_h3_pipeline(full=True, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    hooks = apply_svdquant(pipe.dit, args.state)
    targets = target_linears(pipe.dit)
    if len(hooks) != 200 or len(targets) != 200:
        raise RuntimeError("wrong target linear count")
    recorder = ScaleRecorder(raw_path)
    for (name, _), runtime in zip(targets, hooks, strict=True):
        runtime.act.scale_observer = recorder.observer(name)
    print(f"generation starting: prompt_id={args.prompt_id}, seed={row['seed']}", flush=True)
    try:
        video, audio = pipe(prompt=row["prompt"], seed=int(row["seed"]), height=480,
                            width=832, num_frames=124, num_inference_steps=20,
                            cfg_scale=1.0, rand_device="cpu", tiled=True)
    finally:
        recorder.flush()
    if len(video) != 124 or any(h.act.calls != 20 for h in hooks):
        raise RuntimeError("incomplete generation or activation-hook coverage")
    write_video_audio(video=video, audio=audio, output_path=str(video_path), fps=24,
                      audio_sample_rate=pipe.audio_vae.sample_rate)
    report = {
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "state": str(args.state.resolve()), "prompt_id": args.prompt_id,
        "seed": int(row["seed"]), "prompt_source": str(Path(os.environ["MINIMAX_H3_PROMPT_FILE"])),
        "settings": {"height": 480, "width": 832, "frames": 124, "steps": 20,
                     "fps": 24, "cfg_scale": 1.0, "tiled": True},
        "video": str(video_path), "raw_calls": str(raw_path),
        "statistics": summarize(raw_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"COMPLETE: {report_path} and {video_path}", flush=True)


if __name__ == "__main__":
    main()
