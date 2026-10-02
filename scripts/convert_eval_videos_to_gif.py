#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOTS = (
    Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-rcm-wan"),
    Path("/home/admin/workspace/aop_lab/app_data/videos/svdquant-videoeval-minimax-h3"),
)


def output_stem(record: dict) -> str:
    collection = record["collection"]
    source = Path(record["source_relative_path"])
    sidecar = record.get("sidecar") or {}
    case_id = sidecar.get("case_id", source.parts[0])
    variant = sidecar.get("variant", source.name.split("_id", 1)[0])

    if collection in {"h3_vbench51_r32", "h3_standard_r32"} and variant == "svdquant":
        variant = "svdquant_r32"
    elif collection in {"h3_vbench51_r64", "h3_standard_r64"} and variant == "svdquant":
        variant = "svdquant_r64"
    elif collection == "rcm_real_nvfp4_g10_r64_vbench51" and variant == "nvfp4_svdquant":
        variant = "nvfp4_svdquant_g10_r64"
    elif collection == "rcm_real_nvfp4_g20_r32_vbench51" and variant == "nvfp4_svdquant":
        variant = "nvfp4_svdquant_g20_r32"

    return f"{case_id}_{variant}"


def collect_jobs(roots: tuple[Path, ...], suffix: str) -> list[tuple[Path, Path]]:
    jobs: dict[Path, Path] = {}
    for root in roots:
        rows = [json.loads(line) for line in (root / "metadata" / "videos.jsonl").read_text().splitlines() if line]
        for record in rows:
            source = root / record["published_path"]
            target = root / "videos" / f"{output_stem(record)}{suffix}"
            previous = jobs.get(target)
            if previous is not None and previous != source:
                raise ValueError(f"conflicting sources for {target}: {previous} vs {source}")
            jobs[target] = source
    return sorted(jobs.items(), key=lambda item: str(item[0]))


def convert(source: Path, target: Path, ffmpeg: str) -> tuple[bool, str]:
    if target.is_file() and target.stat().st_size > 0:
        return True, "skipped"

    partial = target.with_suffix(".gif.part")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        "fps=10,scale=320:-2:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop",
        "0",
        "-f",
        "gif",
        str(partial),
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        return False, result.stderr.strip() or f"ffmpeg exited {result.returncode}"
    os.replace(partial, target)
    return True, "converted"


def link_original(source: Path, target: Path) -> tuple[bool, str]:
    if target.exists():
        if target.samefile(source):
            return True, "skipped"
        return False, f"existing target is not source: {target}"
    os.link(source, target)
    return True, "linked"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materialize-mp4", action="store_true")
    parser.add_argument("--delete-gifs", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.delete_gifs and not args.materialize_mp4:
        parser.error("--delete-gifs requires --materialize-mp4")
    if not args.materialize_mp4 and (not args.ffmpeg or not Path(args.ffmpeg).is_file()):
        parser.error("a valid --ffmpeg path is required for GIF conversion")

    suffix = ".mp4" if args.materialize_mp4 else ".gif"
    jobs = collect_jobs(ROOTS, suffix)
    if args.materialize_mp4:
        existing = sum(target.exists() for target, _ in jobs)
        print(f"mp4_targets={len(jobs)} existing={existing} pending={len(jobs) - existing}", flush=True)
        if args.dry_run:
            return 0

        failures: list[tuple[Path, str]] = []
        for completed, (target, source) in enumerate(jobs, start=1):
            try:
                ok, status = link_original(source, target)
            except Exception as error:
                ok, status = False, repr(error)
            if not ok:
                failures.append((target, status))
            if completed % 10 == 0 or not ok or completed == len(jobs):
                print(f"linked={completed}/{len(jobs)} failed={len(failures)} latest={target.name} {status}", flush=True)

        if failures:
            for target, error in failures:
                print(f"FAILED {target}: {error}", file=sys.stderr, flush=True)
            return 1
        if args.delete_gifs:
            gifs = [gif for root in ROOTS for gif in (root / "videos").glob("*.gif")]
            for gif in gifs:
                gif.unlink()
            print(f"deleted_gifs={len(gifs)}", flush=True)
        print("ALL DONE", flush=True)
        return 0

    pending = [(target, source) for target, source in jobs if not (target.is_file() and target.stat().st_size > 0)]
    print(f"gif_targets={len(jobs)} existing={len(jobs) - len(pending)} pending={len(pending)}", flush=True)
    if args.dry_run:
        return 0

    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(convert, source, target, args.ffmpeg): target for target, source in pending}
        for future in as_completed(futures):
            target = futures[future]
            completed += 1
            try:
                ok, status = future.result()
            except Exception as error:
                ok, status = False, repr(error)
            if not ok:
                failures.append((target, status))
            if completed % 10 == 0 or not ok or completed == len(pending):
                print(f"completed={completed}/{len(pending)} failed={len(failures)} latest={target.name} {status}", flush=True)

    if failures:
        for target, error in failures:
            print(f"FAILED {target}: {error}", file=sys.stderr, flush=True)
        return 1
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
