#!/usr/bin/env python
"""Aggregate VisionReward shard outputs into per-variant summaries."""
import glob
import json
import os
from collections import defaultdict

RUN_DIR = "/home/admin/workspace/aop_lab/app_data/runs/visionreward-vbench51"

records = []
for p in sorted(glob.glob(os.path.join(RUN_DIR, "scores_shard*.jsonl"))):
    with open(p) as f:
        for line in f:
            records.append(json.loads(line))

by_variant = defaultdict(list)
for r in records:
    by_variant[(r["model"], r["variant"])].append(r)

summary = {}
for (model, variant), rs in sorted(by_variant.items()):
    scores = [r["score"] for r in rs]
    yes_counts = [sum(1 for a in r["answers"] if a == 1) for r in rs]
    summary[f"{model}/{variant}"] = {
        "model": model,
        "variant": variant,
        "n_videos": len(rs),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "mean_yes_ratio": sum(yes_counts) / (len(rs) * rs[0]["questions"]),
    }

with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump({"records": len(records), "variants": summary}, f, indent=2)

with open(os.path.join(RUN_DIR, "scores_all.jsonl"), "w") as f:
    for r in sorted(records, key=lambda x: (x["model"], x["variant"], x["case_id"])):
        f.write(json.dumps(r) + "\n")

print(f"{'model/variant':45s} {'n':>4s} {'mean':>8s} {'min':>8s} {'max':>8s} {'yes%':>6s}")
for k, s in summary.items():
    print(f"{k:45s} {s['n_videos']:4d} {s['mean_score']:8.4f} {s['min_score']:8.4f} "
          f"{s['max_score']:8.4f} {s['mean_yes_ratio']*100:5.1f}%")
print(f"\ntotal records: {len(records)} (expect 561)")
