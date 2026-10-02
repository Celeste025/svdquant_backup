#!/usr/bin/env python
"""VisionReward-Video scoring for the ConvRot NVFP4 variant over the H3 VBench-51 subset."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visionreward_common as vr

VIDEOS_ROOT = "/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3"


def build_tasks():
    tasks = []
    with open(os.path.join(VIDEOS_ROOT, "metadata", "videos.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e["collection"] != "h3_vbench51_convrot":
                continue
            sc = e.get("sidecar") or {}
            if sc.get("variant") != "convrot" or not sc.get("case_id"):
                continue
            tasks.append({
                "model": "minimax_h3",
                "variant": "convrot",
                "case_id": sc["case_id"],
                "sha256": e["sha256"],
                "prompt": sc.get("prompt"),
                "path": os.path.join(VIDEOS_ROOT, "videos", e["sha256"] + ".mp4"),
            })
    return sorted(tasks, key=lambda t: t["case_id"])


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
                done.add(r["case_id"])
    todo = [t for t in tasks if t["case_id"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] tasks={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, tokenizer = vr.load_model()
    t0 = time.time()
    with open(args.output, "a") as f:
        for i, t in enumerate(todo):
            score, answers = vr.score_video(model, tokenizer, t["path"], t["prompt"])
            rec = {"model": "minimax_h3", "variant": "convrot", "case_id": t["case_id"],
                   "sha256": t["sha256"], "score": score, "answers": answers,
                   "questions": len(vr.QUESTIONS)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"[shard {args.shard}] {i+1}/{len(todo)} done ({el/(i+1):.1f}s/video, eta {(len(todo)-i-1)*el/(i+1)/60:.1f} min)", flush=True)
    print(f"[shard {args.shard}] ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
