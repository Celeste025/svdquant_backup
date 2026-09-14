#!/usr/bin/env python3
"""Paired end-to-end test of blockwise-refined LowRank branches in rCM-Wan."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import WanPipeline

from eval_rcm_restore_video_similarity import compare as compare_video
from exp_rcm_blockwise_lowrank_pilot import branch_map
from exp_rcm_qk_bf16_oracle import build_bf16_reference, decode_and_save, tensor_metrics
from exp_rcm_restore_top_fraction_bf16 import CKPT, MODEL, PROMPT, load_helper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINED = ROOT / "results/reports/rcm_blockwise_lowrank_pilot/blocks0_8_16_train4_test4_e10/trained_lowrank_branches.pt"
DEFAULT_OUT = ROOT / "results/samples/rcm_blockwise_lowrank_end2end/airplane_seed303"
CASE_BLOCKS = {
    "nvfp4": (),
    "block0": (0,),
    "block0_8_16": (0, 8, 16),
    "block0_8_16_24": (0, 8, 16, 24),
    "all30": tuple(range(30)),
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def branch_fingerprints(transformer: torch.nn.Module) -> dict[tuple[int, str], tuple[float, float]]:
    result = {}
    for block_index, block in enumerate(transformer.blocks):
        for name, branch in branch_map(block).items():
            tensors = [value.detach().float() for value in branch.state_dict().values()]
            result[(block_index, name)] = (
                sum(float(value.sum()) for value in tensors),
                sum(float(value.square().sum()) for value in tensors),
            )
    if len(result) != 300:
        raise RuntimeError(f"expected 300 LowRank branches, found {len(result)}")
    return result


def install_trained_branches(
    transformer: torch.nn.Module,
    trained: dict[int, dict[str, dict[str, torch.Tensor]]],
    selected_blocks: tuple[int, ...],
) -> dict[str, Any]:
    before = branch_fingerprints(transformer)
    loaded = []
    selected_keys: set[tuple[int, str]] = set()
    for block_index in selected_blocks:
        if block_index not in trained:
            raise RuntimeError(f"trained checkpoint has no block {block_index}")
        live = branch_map(transformer.blocks[block_index])
        saved = trained[block_index]
        if set(live) != set(saved):
            raise RuntimeError(f"block {block_index}: branch names differ: live={sorted(live)}, saved={sorted(saved)}")
        for name, branch in live.items():
            selected_keys.add((block_index, name))
            expected = branch.state_dict()
            state = saved[name]
            if set(expected) != set(state):
                raise RuntimeError(f"block {block_index}.{name}: state keys differ")
            for key in expected:
                if expected[key].shape != state[key].shape:
                    raise RuntimeError(f"block {block_index}.{name}.{key}: shape {state[key].shape} != {expected[key].shape}")
            branch.load_state_dict(state, strict=True)
            loaded.append(f"blocks.{block_index}.{name}")

    after = branch_fingerprints(transformer)
    unexpected = [key for key in before if key not in selected_keys and before[key] != after[key]]
    if unexpected:
        raise RuntimeError(f"non-selected LowRank branches changed: {unexpected[:5]}")
    unchanged_selected = [key for key in selected_keys if before[key] == after[key]]
    if unchanged_selected:
        raise RuntimeError(f"selected trained branches did not change: {unchanged_selected[:5]}")
    return {
        "selected_blocks": list(selected_blocks),
        "loaded_branches": loaded,
        "loaded_count": len(loaded),
        "non_selected_checked": len(before) - len(selected_keys),
        "non_selected_changed": 0,
    }


@torch.inference_mode()
def run_case(
    case: str,
    selected_blocks: tuple[int, ...],
    trained: dict[int, dict[str, dict[str, torch.Tensor]]],
    reference: dict[str, Any],
    output_dir: Path,
    helper: Any,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda")
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    helper.load_quantized_transformer(pipe, CKPT, MODEL)
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    audit = install_trained_branches(pipe.transformer, trained, selected_blocks) if selected_blocks else {
        "selected_blocks": [], "loaded_branches": [], "loaded_count": 0,
        "non_selected_checked": 300, "non_selected_changed": 0,
    }

    embeds = reference["prompt_embeds"].to(device=device, dtype=pipe.transformer.dtype)
    times = torch.tensor(reference["times"], device=device, dtype=torch.float64)
    ones = torch.ones((1,), device=device, dtype=torch.float64)
    latents = reference["initial_latent"].to(device=device, dtype=torch.float64).clone()
    rows = []
    for step, (current, nxt) in enumerate(zip(times[:-1], times[1:], strict=True)):
        timestep = (current.float() * ones * 1000).to(dtype=pipe.transformer.dtype)
        teacher_input = reference["inputs"][step].to(device=device, dtype=torch.float64)
        teacher_velocity = pipe.transformer(
            hidden_states=teacher_input.to(pipe.transformer.dtype), timestep=timestep,
            encoder_hidden_states=embeds, return_dict=False,
        )[0]
        teacher = tensor_metrics(teacher_velocity, reference["velocities"][step])
        if step == 0:
            rollout_velocity = teacher_velocity.to(torch.float64)
        else:
            rollout_velocity = pipe.transformer(
                hidden_states=latents.to(pipe.transformer.dtype), timestep=timestep,
                encoder_hidden_states=embeds, return_dict=False,
            )[0].to(torch.float64)
        rollout = tensor_metrics(rollout_velocity, reference["velocities"][step])
        latents = (1 - nxt) * (latents - current * rollout_velocity) + nxt * reference["noises"][step].to(
            device=device, dtype=torch.float64
        )
        latent = tensor_metrics(latents, reference["latents"][step + 1])
        rows.append({
            "case": case, "step": step, "timestep": float(current),
            **{f"teacher_{key}": value for key, value in teacher.items()},
            **{f"rollout_{key}": value for key, value in rollout.items()},
            **{f"latent_{key}": value for key, value in latent.items()},
        })
        print(
            f"case={case} step={step} teacher_nmse={teacher['nmse']:.6g} "
            f"rollout_nmse={rollout['nmse']:.6g} latent_nmse={latent['nmse']:.6g}", flush=True,
        )

    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    video = output_dir / f"{case}.mp4"
    decode_and_save(pipe, helper, latents, video)
    pipe.to("cpu")
    del pipe, embeds, latents
    gc.collect()
    torch.cuda.empty_cache()
    return video, rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--trained", type=Path, default=DEFAULT_TRAINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_BLOCKS), default=list(CASE_BLOCKS))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    helper = load_helper()
    trained = torch.load(args.trained, map_location="cpu", weights_only=False)
    required_blocks = {block for case in args.cases for block in CASE_BLOCKS[case]}
    missing_blocks = required_blocks - set(trained)
    if missing_blocks:
        raise RuntimeError(f"trained checkpoint is missing requested blocks {sorted(missing_blocks)}; got {sorted(trained)}")


    bf16_video = output_dir / "bf16.mp4"
    reference, _, _ = build_bf16_reference(
        args.prompt, args.seed, args.height, args.width, args.frames, bf16_video, helper
    )
    write_json(output_dir / "run_config.json", {
        "prompt": args.prompt, "seed": args.seed, "height": args.height, "width": args.width,
        "frames": args.frames, "schedule": "rCM TrigFlow->RectifiedFlow, 4 steps, guidance=0",
        "quant_checkpoint": str(CKPT), "trained_lowrank": str(args.trained.resolve()),
        "cases": {case: list(CASE_BLOCKS[case]) for case in args.cases},
        "comparison": "paired prompt embedding, initial latent, per-step noise, and schedule",
    })

    all_rows = []
    videos = {}
    for case in CASE_BLOCKS:
        if case not in args.cases:
            continue
        print(f"[run] {case} blocks={CASE_BLOCKS[case]}", flush=True)
        video, rows, audit = run_case(case, CASE_BLOCKS[case], trained, reference, output_dir, helper)
        videos[case] = video
        all_rows.extend(rows)
        write_json(output_dir / f"{case}_audit.json", audit)
    write_csv(output_dir / "per_step_metrics.csv", all_rows)
    write_json(output_dir / "per_step_metrics.json", {
        "definition": "Teacher uses paired BF16 latent; rollout uses each case's own latent. Latent is measured after paired scheduler update.",
        "rows": all_rows,
    })

    video_rows = []
    for case, video in videos.items():
        print(f"[video metrics] {case}", flush=True)
        metrics = compare_video(bf16_video, video, torch.device("cuda"), include_temporal=False)
        video_rows.append({"case": case, "reference": str(bf16_video), "video": str(video), **metrics})
        torch.cuda.empty_cache()
    write_csv(output_dir / "video_metrics.csv", video_rows)
    write_json(output_dir / "video_metrics.json", {
        "definition": "Full-reference decoded-video metrics against paired BF16; lower MSE/NMSE/MAE/LPIPS and higher PSNR/SSIM are better. Temporal-LPIPS is intentionally excluded.",
        "rows": video_rows,
    })
    print(f"saved end-to-end experiment to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
