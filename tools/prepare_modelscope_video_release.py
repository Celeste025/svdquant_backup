#!/usr/bin/env python3
"""Build immutable, deduplicated ModelScope video-evaluation dataset packages.

Only explicitly listed non-smoke experiment directories are included.  Every
video is content-addressed, while its original case path and sidecar metadata
are retained in ``metadata/videos.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data1/models/svdquant-wjq")


@dataclass(frozen=True)
class Collection:
    name: str
    path: Path
    description: str


DATASETS: dict[str, tuple[Collection, ...]] = {
    "svdquant-videoeval-rcm-wan": (
        Collection("rcm_vbench251_bf16_nvfp4", ROOT / "results/samples/rcm_vbench251_seed0_480p77f_4step", "rCM VBench cases: BF16, NVFP4 W4A4, and SVDQuant."),
        Collection("rcm_int4_g10_vbench51", ROOT / "results/samples/rcm_int4_svdquant_vbench51_seed0_480p77f_4step", "rCM INT4 g10 comparison: BF16, plain INT4, and SVDQuant INT4."),
        Collection("rcm_real_nvfp4_g20_r32_vbench51", ROOT / "results/samples/rcm_real_nvfp4_svdquant_g20_r32_vbench51_seed0_480p77f_4step", "rCM real-NVFP4 grid20/rank32 VBench-51."),
        Collection("rcm_real_nvfp4_g10_r64_vbench51", ROOT / "results/samples/rcm_real_nvfp4_svdquant_g10_r64_vbench51_seed0_480p77f_4step", "rCM real-NVFP4 grid10/rank64 VBench-51."),
    ),
    "svdquant-videoeval-minimax-h3": (
        Collection("h3_vbench51_r32", ROOT / "results/samples/minimax_h3_vbench51_seed0_calibshape", "MiniMax-H3 VBench-51: BF16, plain W4A4, SVDQuant rank32."),
        Collection("h3_vbench51_r64", ROOT / "results/samples/minimax_h3_svdquant_standard_r64_vbench51_seed0_calibshape", "MiniMax-H3 VBench-51: paired BF16/W4A4 and SVDQuant rank64."),
        Collection("h3_standard_r32", ROOT / "results/samples/minimax_h3_svdquant_standard_8p64s", "MiniMax-H3 standard selected p2/p16/p26/p42/p51 comparisons, rank32."),
        Collection("h3_standard_r64", ROOT / "results/samples/minimax_h3_svdquant_standard_r64_8p64s", "MiniMax-H3 available standard rank64 comparison videos."),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sidecar(video: Path) -> dict | None:
    candidate = video.with_suffix(".json")
    if not candidate.is_file():
        return None
    try:
        value = json.loads(candidate.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_dataset(name: str, collections: tuple[Collection, ...], release_root: Path, overwrite: bool) -> Path:
    destination = release_root / name
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite staged dataset {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    records: list[dict] = []
    seen: dict[str, Path] = {}
    for collection in collections:
        if not collection.path.is_dir():
            raise FileNotFoundError(collection.path)
        # Preserve every text metadata file, including the manifest, per-case
        # prompt/status records, and existing metric summaries.  Videos are
        # handled separately and therefore never duplicated here.
        for source in sorted(collection.path.rglob("*")):
            if not source.is_file() or source.suffix.lower() == ".mp4":
                continue
            relative = source.relative_to(collection.path)
            if "smoke" in str(relative).lower():
                continue
            copy(source, destination / "metadata" / collection.name / relative)
        for video in sorted(collection.path.rglob("*.mp4")):
            digest = sha256(video)
            published = Path("videos") / f"{digest}.mp4"
            target = destination / published
            if digest not in seen:
                copy(video, target)
                seen[digest] = target
            records.append({
                "collection": collection.name,
                "description": collection.description,
                "source_relative_path": str(video.relative_to(collection.path)),
                "published_path": str(published),
                "sha256": digest,
                "bytes": video.stat().st_size,
                "sidecar": sidecar(video),
            })
    metadata = destination / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "videos.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset": name,
        "collections": [{"name": x.name, "description": x.description, "source_paths_removed": True} for x in collections],
        "video_records": len(records),
        "unique_videos": len(seen),
        "smoke_or_convrot_included": False,
    }
    (destination / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        f"# {name}\n\n"
        "Core non-smoke SVDQuant evaluation videos. Videos are content-addressed under `videos/`; "
        "use `metadata/videos.jsonl` to map each original case, prompt/status sidecar, and variant to a video. "
        "The `metadata/` tree retains source manifests and metric records.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (destination / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(destination)}\n" for path in files), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DATA / "releases/modelscope/v0.1.0/datasets")
    parser.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name, collections in DATASETS.items():
        if args.dataset and name not in args.dataset:
            continue
        print(write_dataset(name, collections, args.release_root, args.overwrite), flush=True)


if __name__ == "__main__":
    main()
