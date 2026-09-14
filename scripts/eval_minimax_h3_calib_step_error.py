#!/usr/bin/env python3
"""Teacher-forced complete-DiT BF16/SVDQuant error on cached H3 calibration calls."""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch

from eval_minimax_h3_paired_step_error import apply_svdquant, metrics, split_output
from minimax_h3_svdquant_common import load_h3_pipeline, tree_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_calls(cache_dir: Path):
    calls = []
    for manifest_path in sorted(cache_dir.glob("p*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for entry in sorted(manifest["samples"], key=lambda x: int(x["step"])):
            item = torch.load(manifest_path.parent / entry["file"], map_location="cpu", weights_only=False)
            calls.append((item["meta"], item["input_args"], item["input_kwargs"]))
    if len(calls) != 64:
        raise RuntimeError(f"expected 64 cached calls, got {len(calls)}")
    return calls


@torch.inference_mode()
def main():
    args = parse_args()
    calls = load_calls(args.cache_dir)
    print("running BF16 references for 64 cached calibration calls", flush=True)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    refs = []
    for i, (_meta, input_args, input_kwargs) in enumerate(calls, 1):
        video, audio = split_output(pipe.dit(*tree_device(input_args, "cuda"), **tree_device(input_kwargs, "cuda")))
        refs.append((video.cpu(), audio.cpu()))
        print(f"BF16 {i}/64", flush=True)
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    print("installing SVDQuant and replaying the same 64 calls", flush=True)
    pipe = load_h3_pipeline(full=False, reserve_gib=35.0)
    pipe.load_models_to_device(["dit"])
    hooks = apply_svdquant(pipe.dit, args.state)
    details = []
    for i, ((meta, input_args, input_kwargs), (ref_video, ref_audio)) in enumerate(zip(calls, refs, strict=True), 1):
        video, audio = split_output(pipe.dit(*tree_device(input_args, "cuda"), **tree_device(input_kwargs, "cuda")))
        record = {"prompt_id": int(meta["prompt_id"]), "seed": int(meta["seed"]), "step": int(meta["step"]),
                  "timestep": float(input_kwargs["unique_timesteps"].flatten()[0]),
                  "video": metrics(video, ref_video.to("cuda")), "audio": metrics(audio, ref_audio.to("cuda"))}
        details.append(record)
        print(f"SVDQ {i}/64 p{record['prompt_id']} s{record['step']:02d} "
              f"nmse={record['video']['nmse']:.6g}", flush=True)
    if any(h.act.calls != 64 for h in hooks):
        raise RuntimeError("one or more quantization hooks did not see every calibration call")
    grouped = defaultdict(list)
    for x in details:
        grouped[x["step"]].append(x)
    summary = []
    for step, rows in sorted(grouped.items()):
        # Energy-weighted NMSE is the appropriate aggregate, not an unweighted mean.
        nums = [r["video"]["mse"] for r in rows]
        dens = [r["video"]["mse"] / max(r["video"]["nmse"], 1e-30) for r in rows]
        summary.append({"step": step, "timestep_values": sorted({r["timestep"] for r in rows}),
                        "video_nmse_mean": sum(r["video"]["nmse"] for r in rows) / len(rows),
                        "video_nmse_energy_weighted": sum(nums) / sum(dens),
                        "video_nmse_min": min(r["video"]["nmse"] for r in rows),
                        "video_nmse_max": max(r["video"]["nmse"] for r in rows), "samples": len(rows)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"definition": "complete DiT teacher-forced output error; BF16 and SVDQuant receive identical cached calibration-call inputs", "details": details, "by_step": summary}, indent=2))
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
