#!/usr/bin/env python
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_visionreward_q64_all as runner
import visionreward_q64 as vr


def load_records(output_root, tasks):
    records = {}
    for task in tasks:
        path = runner.record_path(output_root, task["task_id"])
        if not path.is_file():
            raise RuntimeError(f"missing score record: {path}")
        record = json.loads(path.read_text())
        if not runner.valid_record(record, task):
            raise RuntimeError(f"invalid score record: {path}")
        records[task["task_id"]] = record
    return records


def topic_statistics(records):
    binary = np.array([record["binary_yes"] for record in records], dtype=float)
    probs = np.array([record["p_yes_vs_no"] for record in records], dtype=float)
    topics = {}
    for name, indices in vr.TOPICS:
        topics[name] = {
            "question_ids": list(range(indices.start + 1, indices.stop + 1)),
            "binary_yes_rate": float(binary[:, indices].mean()),
            "mean_p_yes": float(probs[:, indices].mean()),
        }
    return topics


def aggregate(rows):
    score_records = [row["score_record"] for row in rows]
    return {
        "n_mappings": len(rows),
        "n_unique_videos": len({row["score_task_id"] for row in rows}),
        "unweighted_64q_binary_yes_rate": float(
            np.mean([record["unweighted_64q_binary_yes_rate"] for record in score_records])
        ),
        "unweighted_64q_mean_p_yes": float(
            np.mean([record["unweighted_64q_mean_p_yes"] for record in score_records])
        ),
        "topics": topic_statistics(score_records),
    }


def summarize_group(rows, key_fn):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return {key: aggregate(value) for key, value in sorted(grouped.items())}


def export_score_jsonl(output_root, tasks, records):
    all_rows = []
    shard_rows = {0: [], 1: [], 2: []}
    for index, task in enumerate(tasks):
        record = records[task["task_id"]]
        all_rows.append(record)
        shard_rows[index % 3].append(record)
    runner.atomic_write_jsonl(output_root / "scores_q64_all.jsonl", all_rows)
    for shard, rows in shard_rows.items():
        runner.atomic_write_jsonl(output_root / f"scores_q64_shard{shard}.jsonl", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=runner.DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()
    output_root = args.output_root

    mappings, tasks = runner.load_prepared_tasks(output_root)
    if len(mappings) != runner.EXPECTED_MAPPINGS or len(tasks) != runner.EXPECTED_UNIQUE_TASKS:
        raise RuntimeError("unexpected prepared Q64 input counts")
    records = load_records(output_root, tasks)
    if len(records) != runner.EXPECTED_UNIQUE_TASKS:
        raise RuntimeError("incomplete Q64 score set")

    decorated = []
    for mapping in mappings:
        score_record = records.get(mapping["score_task_id"])
        if score_record is None:
            raise RuntimeError(f"unmapped score task: {mapping['score_task_id']}")
        decorated.append({**mapping, "score_record": score_record})
    if len(decorated) != runner.EXPECTED_MAPPINGS:
        raise RuntimeError("mapping join count mismatch")

    export_score_jsonl(output_root, tasks, records)
    summary = {
        "schema_version": "visionreward-q64/v1",
        "metric_label": vr.METRIC_LABEL,
        "official_weighted_score_available": False,
        "question_manifest_sha256": vr.QUESTION_MANIFEST_SHA256,
        "question_count": 64,
        "n_mappings": len(decorated),
        "n_unique_videos": len(records),
        "aggregation": {
            "binary": "unweighted mean of binary Yes across full 64-question checklist",
            "probability": "unweighted mean of P(Yes | Yes, No) across full 64-question checklist",
        },
        "by_collection_source_variant": summarize_group(
            decorated,
            lambda row: f"{row['collection']}/{row['source_variant']}",
        ),
        "by_model_variant": summarize_group(
            decorated,
            lambda row: f"{row['model']}/{row['variant']}",
        ),
        "by_scope_model_variant": summarize_group(
            decorated,
            lambda row: f"{row['scope']}/{row['model']}/{row['variant']}",
        ),
    }
    runner.atomic_write_json(output_root / "summary_q64.json", summary)

    print(f"Q64 records: {len(records)} unique videos, {len(decorated)} mappings")
    print(f"{'model/variant':40s} {'maps':>6s} {'unique':>7s} {'binary_yes':>11s} {'mean_p_yes':>11s}")
    for key, values in summary["by_model_variant"].items():
        print(
            f"{key:40s} {values['n_mappings']:6d} {values['n_unique_videos']:7d} "
            f"{values['unweighted_64q_binary_yes_rate']:11.4f} "
            f"{values['unweighted_64q_mean_p_yes']:11.4f}"
        )


if __name__ == "__main__":
    main()
