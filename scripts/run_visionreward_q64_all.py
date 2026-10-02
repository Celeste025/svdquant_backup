#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import torch

import visionreward_q64 as vr

DATA_ROOT = Path("/home/admin/workspace/aop_lab/app_data")
VIDEOS_ROOT = DATA_ROOT / "videos"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "runs" / "visionreward-q64"
DATASETS = (
    "svdquant-videoeval-rcm-wan",
    "svdquant-videoeval-minimax-h3",
)
RCM_COLLECTIONS = {
    "rcm_vbench251_bf16_nvfp4",
    "rcm_int4_g10_vbench51",
    "rcm_real_nvfp4_g20_r32_vbench51",
    "rcm_real_nvfp4_g10_r64_vbench51",
}
H3_VBENCH_COLLECTIONS = {
    "h3_vbench51_r32",
    "h3_vbench51_r64",
    "h3_vbench51_convrot",
}
H3_STANDARD_COLLECTIONS = {"h3_standard_r32", "h3_standard_r64"}
STANDARD_PROMPT_IDS = {"p2", "p16", "p26", "p42", "p51"}
EXPECTED_MAPPINGS = 1037
EXPECTED_UNIQUE_TASKS = 680


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def load_standard_prompts():
    csv_path = (
        VIDEOS_ROOT
        / "svdquant-videoeval-minimax-h3"
        / "metadata"
        / "h3_standard_r32"
        / "mjvideo_prompt_metrics.csv"
    )
    prompts = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = row["case_id"]
            prompt = row["prompt"]
            if case_id in prompts and prompts[case_id] != prompt:
                raise RuntimeError(f"conflicting standard prompt for {case_id}")
            prompts[case_id] = prompt
    if set(prompts) != STANDARD_PROMPT_IDS:
        raise RuntimeError(f"unexpected standard prompt ids: {sorted(prompts)}")
    return prompts


def logical_variant(dataset, collection, source_variant):
    if dataset == "svdquant-videoeval-rcm-wan":
        mapping = {
            ("rcm_vbench251_bf16_nvfp4", "bf16"): "bf16",
            ("rcm_vbench251_bf16_nvfp4", "nvfp4"): "nvfp4",
            ("rcm_vbench251_bf16_nvfp4", "nvfp4_svdquant"): "nvfp4_svdquant",
            ("rcm_int4_g10_vbench51", "bf16"): "bf16",
            ("rcm_int4_g10_vbench51", "int4_plain"): "int4_plain",
            ("rcm_int4_g10_vbench51", "int4_svdquant"): "int4_svdquant",
            ("rcm_real_nvfp4_g20_r32_vbench51", "bf16"): "bf16",
            ("rcm_real_nvfp4_g20_r32_vbench51", "nvfp4_svdquant"): "nvfp4_svdquant_g20_r32",
            ("rcm_real_nvfp4_g10_r64_vbench51", "bf16"): "bf16",
            ("rcm_real_nvfp4_g10_r64_vbench51", "nvfp4_svdquant"): "nvfp4_svdquant_g10_r64",
        }
        return "rcm_wan", mapping[(collection, source_variant)]

    if collection in {"h3_vbench51_r32", "h3_standard_r32"}:
        mapping = {"bf16": "bf16", "w4a4": "w4a4", "svdquant": "svdquant_r32"}
    elif collection in {"h3_vbench51_r64", "h3_standard_r64"}:
        mapping = {"bf16": "bf16", "w4a4": "w4a4", "svdquant": "svdquant_r64"}
    elif collection == "h3_vbench51_convrot":
        mapping = {"bf16": "bf16", "w4a4": "w4a4", "convrot": "convrot"}
    else:
        raise RuntimeError(f"unknown H3 collection {collection}")
    return "minimax_h3", mapping[source_variant]


def scope_for(dataset, collection):
    if dataset == "svdquant-videoeval-rcm-wan":
        return "rcm_vbench"
    if collection in H3_STANDARD_COLLECTIONS:
        return "h3_standard"
    if collection == "h3_vbench51_convrot":
        return "h3_vbench_convrot"
    return "h3_vbench"


def build_mappings():
    standard_prompts = load_standard_prompts()
    mappings = []

    for dataset in DATASETS:
        metadata_path = VIDEOS_ROOT / dataset / "metadata" / "videos.jsonl"
        with metadata_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                collection = record["collection"]
                source_path = record["source_relative_path"]
                sidecar = record.get("sidecar") or {}

                if dataset == "svdquant-videoeval-rcm-wan":
                    if collection not in RCM_COLLECTIONS:
                        continue
                    case_id = sidecar["case_id"]
                    source_variant = sidecar["variant"]
                    prompt = sidecar["prompt"]
                    prompt_source = "sidecar.prompt"
                elif collection in H3_VBENCH_COLLECTIONS:
                    case_id = sidecar["case_id"]
                    source_variant = sidecar["variant"]
                    prompt = sidecar["prompt"]
                    prompt_source = "sidecar.prompt"
                elif collection in H3_STANDARD_COLLECTIONS:
                    prompt_id = Path(source_path).parts[0]
                    if prompt_id not in STANDARD_PROMPT_IDS:
                        continue
                    case_id = prompt_id
                    source_variant = Path(source_path).stem.split("_id", 1)[0]
                    prompt = standard_prompts[prompt_id]
                    prompt_source = "h3_standard_r32/mjvideo_prompt_metrics.csv"
                else:
                    continue

                model, variant = logical_variant(dataset, collection, source_variant)
                sha256 = record["sha256"]
                prompt_sha256 = sha256_text(prompt)
                video_path = VIDEOS_ROOT / dataset / "videos" / f"{sha256}.mp4"
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                mapping_id = sha256_text(
                    "\0".join((dataset, collection, source_path, sha256, prompt_sha256))
                )
                score_task_id = sha256_text(
                    "\0".join((sha256, prompt_sha256, vr.QUESTION_MANIFEST_SHA256))
                )
                mappings.append(
                    {
                        "mapping_id": mapping_id,
                        "dataset": dataset,
                        "collection": collection,
                        "scope": scope_for(dataset, collection),
                        "model": model,
                        "variant": variant,
                        "source_variant": source_variant,
                        "case_id": case_id,
                        "source_relative_path": source_path,
                        "sha256": sha256,
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256,
                        "prompt_source": prompt_source,
                        "video_path": str(video_path),
                        "score_task_id": score_task_id,
                    }
                )

    mappings.sort(key=lambda row: row["mapping_id"])
    if len(mappings) != EXPECTED_MAPPINGS:
        raise RuntimeError(f"expected {EXPECTED_MAPPINGS} mappings, got {len(mappings)}")
    return mappings


def unique_tasks(mappings):
    by_sha256 = {}
    tasks = {}
    for mapping in mappings:
        prompt_sha256 = mapping["prompt_sha256"]
        existing_prompt = by_sha256.setdefault(mapping["sha256"], prompt_sha256)
        if existing_prompt != prompt_sha256:
            raise RuntimeError(f"same video sha256 has conflicting prompts: {mapping['sha256']}")
        task_id = mapping["score_task_id"]
        task = {
            "task_id": task_id,
            "sha256": mapping["sha256"],
            "prompt": mapping["prompt"],
            "prompt_sha256": prompt_sha256,
            "video_path": mapping["video_path"],
        }
        if task_id in tasks and tasks[task_id] != task:
            raise RuntimeError(f"conflicting task definition for {task_id}")
        tasks[task_id] = task
    ordered = sorted(tasks.values(), key=lambda task: task["task_id"])
    if len(ordered) != EXPECTED_UNIQUE_TASKS:
        raise RuntimeError(f"expected {EXPECTED_UNIQUE_TASKS} unique tasks, got {len(ordered)}")
    return ordered


def question_manifest():
    return {
        "schema_version": "visionreward-q64/v1",
        "metric_label": vr.METRIC_LABEL,
        "question_manifest_sha256": vr.QUESTION_MANIFEST_SHA256,
        "questions": [
            {"id": index, "text": question}
            for index, question in enumerate(vr.QUESTIONS, start=1)
        ],
        "topics": [
            {"name": name, "question_ids": list(range(indices.start + 1, indices.stop + 1))}
            for name, indices in vr.TOPICS
        ],
        "aggregation": {
            "binary": "unweighted mean of 64 binary yes values",
            "probability": "unweighted mean of 64 P(Yes | Yes, No) values",
            "official_weighted_score_available": False,
            "reason": "the official release provides 29 weights only; no official 64-question weight vector is published",
        },
    }


def prepare(output_root):
    output_root = Path(output_root)
    manifest_paths = [
        output_root / "question_manifest.json",
        output_root / "run_manifest.json",
        output_root / "mappings.jsonl",
    ]
    existing = [path.exists() for path in manifest_paths]
    if any(existing) and not all(existing):
        raise RuntimeError(f"partial Q64 manifest state in {output_root}; resolve it before preparing")

    mappings = build_mappings()
    tasks = unique_tasks(mappings)
    if all(existing):
        run_manifest = json.loads((output_root / "run_manifest.json").read_text())
        if run_manifest["question_manifest_sha256"] != vr.QUESTION_MANIFEST_SHA256:
            raise RuntimeError("existing Q64 output uses a different question manifest")
        if run_manifest["mapping_count"] != len(mappings) or run_manifest["unique_task_count"] != len(tasks):
            raise RuntimeError("existing Q64 output has different input counts")
        return mappings, tasks

    (output_root / "records").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "question_manifest.json", question_manifest())
    atomic_write_jsonl(output_root / "mappings.jsonl", mappings)
    atomic_write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": "visionreward-q64/v1",
            "metric_label": vr.METRIC_LABEL,
            "model_path": vr.MODEL_PATH,
            "question_manifest_sha256": vr.QUESTION_MANIFEST_SHA256,
            "question_count": len(vr.QUESTIONS),
            "frame_sampling": "chat strategy, up to 24 frames",
            "binary_rule": "first generated token decodes exactly to yes",
            "probability_rule": "two-token softmax over Yes and No first token logits",
            "mapping_count": len(mappings),
            "unique_task_count": len(tasks),
            "num_shards": 3,
            "datasets": list(DATASETS),
        },
    )
    return mappings, tasks


def load_prepared_tasks(output_root):
    output_root = Path(output_root)
    required = [
        output_root / "question_manifest.json",
        output_root / "run_manifest.json",
        output_root / "mappings.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Q64 output is not prepared; missing: {missing}")
    run_manifest = json.loads((output_root / "run_manifest.json").read_text())
    if run_manifest["question_manifest_sha256"] != vr.QUESTION_MANIFEST_SHA256:
        raise RuntimeError("question manifest mismatch")
    if run_manifest["question_count"] != 64 or run_manifest["num_shards"] != 3:
        raise RuntimeError("unexpected prepared Q64 run manifest")
    with (output_root / "mappings.jsonl").open() as handle:
        mappings = [json.loads(line) for line in handle]
    return mappings, unique_tasks(mappings)


def record_path(output_root, task_id):
    return Path(output_root) / "records" / f"{task_id}.json"


def valid_record(record, task):
    if record.get("schema_version") != "visionreward-q64/v1":
        return False
    if record.get("metric_label") != vr.METRIC_LABEL:
        return False
    if record.get("task_id") != task["task_id"]:
        return False
    if record.get("sha256") != task["sha256"] or record.get("prompt_sha256") != task["prompt_sha256"]:
        return False
    if record.get("question_manifest_sha256") != vr.QUESTION_MANIFEST_SHA256:
        return False
    if record.get("questions") != 64:
        return False
    for key in ("generated_text", "binary_yes", "p_yes_vs_no"):
        if len(record.get(key, [])) != 64:
            return False
    return all(
        isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
        for value in record["p_yes_vs_no"]
    )


def completed_task_ids(output_root, tasks):
    completed = set()
    for task in tasks:
        path = record_path(output_root, task["task_id"])
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if not valid_record(record, task):
            raise RuntimeError(f"invalid completed record: {path}")
        completed.add(task["task_id"])
    return completed


def run_worker(output_root, gpu, shard, num_shards, task_id=None, limit=0):
    if num_shards != 3:
        raise RuntimeError("Q64 evaluation uses a fixed three-shard partition")
    if shard < 0 or shard >= num_shards:
        raise ValueError(f"invalid shard {shard}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VisionReward Q64 evaluation")

    _, tasks = load_prepared_tasks(output_root)
    shard_tasks = [task for index, task in enumerate(tasks) if index % num_shards == shard]
    if task_id is not None:
        shard_tasks = [task for task in shard_tasks if task["task_id"] == task_id]
        if not shard_tasks:
            raise RuntimeError(f"task {task_id} is not assigned to shard {shard}")
    completed = completed_task_ids(output_root, shard_tasks)
    todo = [task for task in shard_tasks if task["task_id"] not in completed]
    if limit:
        todo = todo[:limit]
    print(
        f"[shard {shard}/{num_shards}] tasks={len(shard_tasks)} completed={len(completed)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return

    atomic_write_json(
        Path(output_root) / f"worker_shard{shard}.json",
        {
            "physical_gpu": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": "cuda:0",
            "shard": shard,
            "num_shards": num_shards,
            "question_manifest_sha256": vr.QUESTION_MANIFEST_SHA256,
        },
    )
    model, tokenizer = vr.load_model("cuda:0")
    print(f"[shard {shard}] model loaded", flush=True)

    started = time.monotonic()
    for index, task in enumerate(todo, start=1):
        result = vr.score_video_q64(model, tokenizer, task["video_path"], task["prompt"], "cuda:0")
        record = {
            "schema_version": "visionreward-q64/v1",
            "metric_label": vr.METRIC_LABEL,
            "task_id": task["task_id"],
            "sha256": task["sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "question_manifest_sha256": vr.QUESTION_MANIFEST_SHA256,
            "questions": 64,
            **result,
        }
        atomic_write_json(record_path(output_root, task["task_id"]), record)
        if index % 5 == 0 or index == len(todo):
            elapsed = time.monotonic() - started
            per_video = elapsed / index
            eta_minutes = (len(todo) - index) * per_video / 60
            print(
                f"[shard {shard}] {index}/{len(todo)} done ({per_video:.1f}s/video, eta {eta_minutes:.1f} min)",
                flush=True,
            )
    print(f"[shard {shard}] ALL DONE in {(time.monotonic() - started) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--task-id")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.prepare == args.worker:
        parser.error("provide exactly one of --prepare or --worker")
    if args.prepare:
        mappings, tasks = prepare(args.output_root)
        print(f"prepared mappings={len(mappings)} unique_tasks={len(tasks)} root={args.output_root}")
        return
    if args.gpu is None or args.shard is None:
        parser.error("--worker requires --gpu and --shard")
    run_worker(args.output_root, args.gpu, args.shard, args.num_shards, args.task_id, args.limit)


if __name__ == "__main__":
    main()
