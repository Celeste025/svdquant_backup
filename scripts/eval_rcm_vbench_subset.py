#!/usr/bin/env python3
"""Evaluate paired rCM-Wan videos on a selected VBench quality subset.

The generated videos use one directory per VBench prompt and one MP4 per model.
This utility keeps the original prompt-to-dimension assignment instead of treating
every video as belonging to every requested VBench dimension.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

# The legacy VBench MUSIQ dependency pins imgaug, whose NumPy-1.x
# ``sctypes`` lookup was removed in NumPy 2.  Keep this compatibility shim
# inside the standalone evaluator; it does not affect generation/PTQ.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_],
    }

# VBench's official AMT checkpoint is an OrderedDict serialized by an older
# PyTorch release. PyTorch >=2.6 defaults torch.load(..., weights_only=True),
# which rejects that trusted checkpoint before VBench can read its state dict.
# Scope the compatibility override to this standalone evaluator only.
_torch_load = torch.load


def _load_trusted_vbench_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _load_trusted_vbench_checkpoint

from vbench.distributed import print0
from vbench.utils import init_submodules, save_json


DEFAULT_DIMENSIONS = (
    "temporal_flickering",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-metadata",
        type=Path,
        help=(
            "Optional VBench prompt metadata containing auxiliary_info. Required "
            "for dimensions such as scene when evaluating the pilot manifest."
        ),
    )
    parser.add_argument("--models", nargs="+", default=["bf16", "nvfp4", "all30"])
    parser.add_argument("--dimensions", nargs="+", default=list(DEFAULT_DIMENSIONS))
    return parser.parse_args()


def main():
    args = parse_args()
    selection_doc = json.loads(args.selection.read_text())
    # The older rCM subset files are a list with ``vbench_index``.  The
    # VBench-251 pilot stores selected cases under ``pilot_cases`` and gives
    # each an explicit stable ``case_id``.  Keep both formats compatible.
    is_pilot_manifest = isinstance(selection_doc, dict) and ("pilot_cases" in selection_doc or "cases" in selection_doc)
    if is_pilot_manifest:
        # rCM stores its selected subset as ``pilot_cases``; the H3 runner uses
        # the unambiguous name ``cases`` for its fixed 51-case manifest.
        selection = selection_doc.get("pilot_cases", selection_doc.get("cases"))
    else:
        selection = selection_doc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = tuple(args.dimensions)
    prompt_metadata = {}
    if args.prompt_metadata:
        for entry in json.loads(args.prompt_metadata.read_text()):
            # The rCM prompt file uses ``prompt``/``dimensions`` while VBench's
            # evaluator consumes ``prompt_en``/``dimension``.  We only carry
            # auxiliary labels; videos and evaluated dimensions still come from
            # the fixed pilot manifest.
            prompt = entry.get("prompt", entry.get("prompt_en"))
            if "auxiliary_info" in entry and prompt:
                prompt_metadata[prompt] = entry["auxiliary_info"]

    manifests = {}
    for model in args.models:
        entries = []
        for item in selection:
            if is_pilot_manifest:
                video = (args.generated_dir / "cases" / item["case_id"] / f"{model}.mp4").resolve()
            else:
                prompt_id = f"{item['vbench_index']:04d}"
                video = (args.generated_dir / prompt_id / f"{model}.mp4").resolve()
            if not video.is_file():
                raise FileNotFoundError(video)
            item_dims = [d for d in item["dimensions"] if d in dimensions]
            if item_dims:
                entry = {
                    "prompt_en": item["prompt"],
                    "dimension": item_dims,
                    "video_list": [str(video)],
                }
                auxiliary_info = prompt_metadata.get(item["prompt"])
                if auxiliary_info:
                    entry["auxiliary_info"] = auxiliary_info
                entries.append(entry)
        manifest = args.output_dir / f"{model}_full_info.json"
        save_json(entries, manifest)
        manifests[model] = manifest

    device = torch.device("cuda")
    submodules = init_submodules(dimensions, local=False, read_frame=False)
    summary_rows = []
    for model, manifest in manifests.items():
        result = {}
        for dimension in dimensions:
            module = __import__(f"vbench.{dimension}", fromlist=[f"compute_{dimension}"])
            compute = getattr(module, f"compute_{dimension}")
            print0(f"Evaluating {model}: {dimension}")
            overall, per_video = compute(str(manifest), device, submodules[dimension])
            result[dimension] = {"overall": float(overall), "per_video": per_video}
            summary_rows.append({"model": model, "dimension": dimension, "score": float(overall)})
        save_json(result, args.output_dir / f"{model}_eval_results.json")

    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "dimension", "score"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
