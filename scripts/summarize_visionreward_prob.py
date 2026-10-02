#!/usr/bin/env python
"""Aggregate probability-version VisionReward scores and compare with binary version."""
import glob
import json
import os
from collections import defaultdict
import numpy as np

PROB_DIR = "/home/admin/workspace/aop_lab/app_data/runs/visionreward-prob"
BIN_DIR = "/home/admin/workspace/aop_lab/app_data/runs/visionreward-vbench51"
CONV_DIR = "/home/admin/workspace/aop_lab/app_data/runs/visionreward-convrot"

# 概率版记录
prob_recs = []
for p in sorted(glob.glob(os.path.join(PROB_DIR, "scores_prob_shard*.jsonl"))):
    with open(p) as f:
        for line in f:
            prob_recs.append(json.loads(line))

# 二值版记录（vbench51 561 + convrot 51）
bin_recs = []
for p in [os.path.join(BIN_DIR, "scores_all.jsonl"), os.path.join(CONV_DIR, "scores_convrot.jsonl")]:
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                bin_recs.append(json.loads(line))

by_pv = defaultdict(list)
for r in prob_recs:
    by_pv[(r["model"], r["variant"])].append(r)
by_bv = defaultdict(list)
for r in bin_recs:
    by_bv[(r["model"], r["variant"])].append(r)

# 合并保存概率版全量
with open(os.path.join(PROB_DIR, "scores_prob_all.jsonl"), "w") as f:
    for r in sorted(prob_recs, key=lambda x: (x["model"], x["variant"], x["case_id"])):
        f.write(json.dumps(r) + "\n")

# 对比表
rows = []
for k in sorted(set(by_pv) | set(by_bv)):
    ps = [r["score"] for r in by_pv.get(k, [])]
    bs = [r["score"] for r in by_bv.get(k, [])]
    rows.append((np.mean(ps) if ps else float("nan"), np.mean(bs) if bs else float("nan"), k, len(ps), len(bs)))

rows.sort(reverse=True)
print(f"{'model/variant':40s} {'prob':>8s} {'binary':>8s} {'diff':>8s} {'n_prob':>7s} {'n_bin':>6s}")
summary = {}
for p, b, k, np_, nb in rows:
    print(f"{k[0] + '/' + k[1]:40s} {p:8.4f} {b:8.4f} {p - b:8.4f} {np_:7d} {nb:6d}")
    summary[f"{k[0]}/{k[1]}"] = {"prob_mean": p, "binary_mean": b, "n": np_}

with open(os.path.join(PROB_DIR, "summary_prob_vs_binary.json"), "w") as f:
    json.dump({"records": len(prob_recs), "variants": summary}, f, indent=2)
print(f"\ntotal prob records: {len(prob_recs)}, binary records: {len(bin_recs)}")
