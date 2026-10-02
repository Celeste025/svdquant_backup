#!/usr/bin/env python3
"""Score paired MiniMax-H3 SageAttention samples with official VisionReward."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import visionreward_common as vr

BASE = Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3/metadata/h3_vbench2_100")
DEFAULT_OUTPUT = Path("/home/admin/workspace/aop_lab/app_source/wjq/svdquant_backup/results/h3_sage_attention_visionreward.jsonl")
VARIANTS = ("bf16", "svdquant", "svdquant_sageattn2", "svdquant_sageattn3")


def video_path(base: Path, case_id: str, case_index: int, variant: str) -> Path:
    if variant in {"bf16", "svdquant"}:
        return base / "cases" / case_id / f"{variant}.mp4"
    gpu = case_index // 2
    backend = "sageattn2" if variant == "svdquant_sageattn2" else "sageattn3"
    return base / f"{backend}_cases8" / f"gpu{gpu}" / "cases" / case_id / "svdquant.mp4"


def load_done(output: Path) -> set[tuple[str, str]]:
    if not output.exists():
        return set()
    done = set()
    with output.open() as handle:
        for line in handle:
            row = json.loads(line)
            done.add((row["case_id"], row["variant"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=BASE / "manifest.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"][:args.cases]
    tasks = []
    for index, case in enumerate(cases):
        for variant in VARIANTS:
            path = video_path(BASE, case["case_id"], index, variant)
            if not path.is_file():
                raise FileNotFoundError(path)
            tasks.append({
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "variant": variant,
                "video": str(path),
            })

    done = load_done(args.output)
    todo = [task for task in tasks if (task["case_id"], task["variant"]) not in done]
    print(f"tasks={len(tasks)} completed={len(done)} todo={len(todo)}", flush=True)
    for task in todo:
        print(f"{task['case_id']} {task['variant']} {task['video']}", flush=True)
    if args.dry_run or not todo:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = vr.load_model("cuda")
    print("model loaded", flush=True)
    started = time.monotonic()
    with args.output.open("a") as handle:
        for index, task in enumerate(todo, start=1):
            score, answers = vr.score_video(model, tokenizer, task["video"], task["prompt"], device="cuda")
            row = {**task, "visionreward_weighted_score": score, "answers": answers,
                   "questions": len(vr.QUESTIONS)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.monotonic() - started
            print(f"{index}/{len(todo)} {task['case_id']} {task['variant']} score={score:.6f} "
                  f"({elapsed / index:.1f}s/video)", flush=True)


if __name__ == "__main__":
    main()
