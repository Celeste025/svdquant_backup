#!/usr/bin/env python
"""Probability-weighted VisionReward scoring for all variants (561 vbench51 + 51 convrot)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visionreward_common as vr
import visionreward_prob as vrp

DATA_ROOT = "/home/admin/workspace/aop_lab/app_data"
VIDEOS_ROOT = os.path.join(DATA_ROOT, "videos")

VARIANT_MAP = {
    ("rcm_int4_g10_vbench51", "bf16"): ("rcm_wan", "bf16"),
    ("rcm_vbench251_bf16_nvfp4", "nvfp4"): ("rcm_wan", "nvfp4"),
    ("rcm_vbench251_bf16_nvfp4", "nvfp4_svdquant"): ("rcm_wan", "nvfp4_svdquant"),
    ("rcm_int4_g10_vbench51", "int4_plain"): ("rcm_wan", "int4_plain"),
    ("rcm_int4_g10_vbench51", "int4_svdquant"): ("rcm_wan", "int4_svdquant"),
    ("rcm_real_nvfp4_g20_r32_vbench51", "nvfp4_svdquant"): ("rcm_wan", "nvfp4_svdquant_g20_r32"),
    ("rcm_real_nvfp4_g10_r64_vbench51", "nvfp4_svdquant"): ("rcm_wan", "nvfp4_svdquant_g10_r64"),
    ("h3_vbench51_r32", "bf16"): ("minimax_h3", "bf16"),
    ("h3_vbench51_r32", "svdquant"): ("minimax_h3", "svdquant_r32"),
    ("h3_vbench51_r64", "svdquant"): ("minimax_h3", "svdquant_r64"),
    ("h3_vbench51_r32", "w4a4"): ("minimax_h3", "w4a4"),
    ("h3_vbench51_r64", "w4a4"): ("minimax_h3", "w4a4"),
    ("h3_vbench51_convrot", "convrot"): ("minimax_h3", "convrot"),
    ("h3_vbench51_convrot", "bf16"): ("minimax_h3", "bf16"),
    ("h3_vbench51_convrot", "w4a4"): ("minimax_h3", "w4a4"),
}

DATASETS = ["svdquant-videoeval-rcm-wan", "svdquant-videoeval-minimax-h3"]


def build_tasks():
    entries = []
    for name in DATASETS:
        with open(os.path.join(VIDEOS_ROOT, name, "metadata", "videos.jsonl")) as f:
            for line in f:
                e = json.loads(line)
                sc = e.get("sidecar") or {}
                if sc.get("case_id"):
                    entries.append({"dataset": name, "collection": e["collection"],
                                    "case_id": sc["case_id"], "variant": sc["variant"],
                                    "sha256": e["sha256"], "prompt": sc.get("prompt")})

    base_ids = {e["case_id"] for e in entries if e["collection"] == "rcm_int4_g10_vbench51"}

    tasks = {}
    for e in entries:
        if e["case_id"] not in base_ids:
            continue
        key = VARIANT_MAP.get((e["collection"], e["variant"]))
        if key is None:
            continue
        model, variant = key
        task_key = (model, variant, e["case_id"])
        path = os.path.join(VIDEOS_ROOT, e["dataset"], "videos", e["sha256"] + ".mp4")
        if task_key not in tasks:
            tasks[task_key] = {"model": model, "variant": variant, "case_id": e["case_id"],
                               "sha256": e["sha256"], "prompt": e["prompt"], "path": path}

    return sorted(tasks.values(), key=lambda t: (t["model"], t["variant"], t["case_id"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = build_tasks()
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["model"], r["variant"], r["case_id"]))
    todo = [t for t in tasks if (t["model"], t["variant"], t["case_id"]) not in done]
    print(f"[shard {args.shard}/{args.num_shards}] tasks={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    model, tokenizer = vr.load_model()
    print("model loaded", flush=True)

    t0 = time.time()
    with open(args.output, "a") as f:
        for i, t in enumerate(todo):
            ts = time.time()
            score, probs = vrp.prob_score_video(model, tokenizer, t["path"], t["prompt"])
            rec = {"model": t["model"], "variant": t["variant"], "case_id": t["case_id"],
                   "sha256": t["sha256"], "score": score, "probs_yes": probs,
                   "questions": len(vr.QUESTIONS), "method": "prob"}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"[shard {args.shard}] {i+1}/{len(todo)} done ({el/(i+1):.1f}s/video, eta {(len(todo)-i-1)*el/(i+1)/60:.1f} min)", flush=True)
    print(f"[shard {args.shard}] ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
