#!/usr/bin/env python3
"""Score paired rCM VBench videos with the official MJ-VIDEO-2B reward model."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

ASPECTS = ["alignment", "safety", "fineness", "coherence_consistency", "bias_fairness"]
CRITERIA = [
    "object", "attribute", "actions", "count", "location",
    "crime", "shocking", "disgust", "nsfw_evasive", "nsfw_subtle", "political_sensitivity",
    "human_face_distortion", "human_limb_distortion", "object_distortion", "defocused_blur", "motion_blur",
    "spatial_consistency", "action_continuity", "object_disappearance", "abrupt_background_changes",
    "inconsistent_lighting_shadows", "frame_flickering", "object_drift",
    "race_bias", "age_bias", "education_bias", "job_bias", "gender_bias",
]
ASPECT_TO_CRITERIA = {0: list(range(0, 5)), 1: list(range(5, 11)), 2: list(range(11, 16)), 3: list(range(16, 23)), 4: list(range(23, 28))}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, help="JSON manifest with cases or pilot_cases.")
    p.add_argument("--prompt-source", type=Path, help="JSONL prompt table; use with --prompt-ids to build cases.")
    p.add_argument("--prompt-ids", type=int, nargs="+", help="Prompt IDs selected from --prompt-source.")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", default="OpenGVLab/InternVL2-2B")
    p.add_argument("--mjvideo-repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=["bf16", "int4_plain", "int4_svdquant"])
    p.add_argument("--num-segments", type=int, default=8)
    p.add_argument("--only-complete", action="store_true", help="Keep only cases with every requested variant MP4.")
    p.add_argument("--max-cases", type=int, help="Optional cap after completeness filtering (for smoke tests).")
    p.add_argument("--video-template", default="cases/{case_id}/{variant}.mp4",
                   help="Path relative to --samples; fields come from each manifest case plus {variant}.")
    return p.parse_args()


def video_path(samples, case, variant, template):
    return samples / template.format(**case, variant=variant)


def main():
    args = parse_args()
    # MJ-VIDEO's upstream InternVL forward path calls get_rank() even for
    # single-GPU inference, so mirror the official one-process setup.
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29541")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    sys.path[:0] = [str(args.mjvideo_repo / "scripts"), str(args.mjvideo_repo / "scripts" / "model")]
    from model import InternVLChatRewardModeling, InternVLChatRewardModelingConfig, prepare_chat_input
    # Import only the local video decoder.  Importing data_processor as a package
    # also imports its training dataset and optional cloud-storage dependencies.
    spec = importlib.util.spec_from_file_location("mjvideo_data", args.mjvideo_repo / "scripts" / "data_processor" / "data.py")
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    load_video = data_module.load_video

    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        cases = manifest.get("cases", manifest.get("pilot_cases"))
        if cases is None:
            raise KeyError("Manifest has neither 'cases' nor 'pilot_cases'.")
    elif args.prompt_source and args.prompt_ids:
        wanted = set(args.prompt_ids)
        rows = {}
        with args.prompt_source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prompt_id = int(row["prompt_id"])
                if prompt_id in wanted:
                    rows[prompt_id] = row
        missing = wanted - rows.keys()
        if missing:
            raise KeyError(f"Prompt IDs not found: {sorted(missing)}")
        cases = [{"case_id": f"p{prompt_id}", "prompt_id": prompt_id,
                  "seed": int(rows[prompt_id]["seed"]), "prompt": rows[prompt_id]["prompt"]}
                 for prompt_id in args.prompt_ids]
    else:
        raise ValueError("Supply --manifest, or both --prompt-source and --prompt-ids.")
    if args.only_complete:
        cases = [
            case for case in cases
            if all(video_path(args.samples, case, variant, args.video_template).is_file() for variant in args.variants)
        ]
    if args.max_cases:
        cases = cases[:args.max_cases]
    if not cases:
        raise RuntimeError("No cases selected for evaluation.")
    prior = json.loads(args.output.read_text()) if args.output.exists() else {"model": str(args.model), "num_segments": args.num_segments, "results": {}}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=False)
    # Some InternVL tokenizer revisions return ``(tokenizer, use_fast)``.
    if isinstance(tokenizer, tuple):
        tokenizer = tokenizer[0]
    config = InternVLChatRewardModelingConfig.from_pretrained(
        args.model, num_objectives=28, num_aspects=5, aspect2criteria=ASPECT_TO_CRITERIA,
        gating_temperature=1.0, gating_hidden_dim=1024, gating_n_hidden=3,
    )
    model = InternVLChatRewardModeling(name=str(args.model), config=config)
    model.load_state_dict(load_file(str(args.model / "model.safetensors")), strict=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    model.model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    generation_config = {"max_new_tokens": 1024, "do_sample": False}

    with torch.inference_mode():
        for case in cases:
            target = prior["results"].setdefault(case["case_id"], {"prompt": case["prompt"], "variants": {}})
            for variant in args.variants:
                if variant in target["variants"]:
                    continue
                video = video_path(args.samples, case, variant, args.video_template)
                pixels, patches = load_video(str(video), num_segments=args.num_segments, max_num=1)
                pixels = pixels.to(device="cuda", dtype=torch.bfloat16)
                prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(patches)))
                ids, mask = prepare_chat_input(config, tokenizer, pixels, prefix + case["prompt"], generation_config, device="cuda")
                out = model(pixels, ids, mask)
                target["variants"][variant] = {
                    "video": str(video), "score": float(out.score[0]),
                    "aspects": dict(zip(ASPECTS, map(float, out.aspect_scores[0]))),
                    "criteria": dict(zip(CRITERIA, map(float, out.rewards[0]))),
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(prior, indent=2) + "\n")
                print(f"{case['case_id']} {variant}: {target['variants'][variant]['score']:.6f}", flush=True)


if __name__ == "__main__":
    main()
