#!/usr/bin/env python3
"""Download and checksum-verify one published SVDQuant artifact variant."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "published_artifacts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(variant_dir: Path) -> None:
    sums = variant_dir / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"missing {sums}")
    for line in sums.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = variant_dir / name.strip()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("rcm-wan", "minimax-h3", "flux1"))
    parser.add_argument("variant")
    parser.add_argument("--namespace", help="ModelScope user or organization (required for remote download)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-release", type=Path, help="test against a local staging root instead of downloading")
    parser.add_argument("--verify-only", action="store_true", help="verify a local staging package in place; no copy")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    key = f"{args.family}:{args.variant}"
    spec = registry["artifacts"].get(key)
    if spec is None:
        raise KeyError(f"unknown published artifact {key}")
    repository = registry["repositories"][spec["repository"]]
    relative = Path(spec["path"])
    if args.local_release:
        source = args.local_release / repository / relative
        if args.verify_only:
            target = source
        else:
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        if args.verify_only:
            parser.error("--verify-only requires --local-release")
        if not args.namespace:
            parser.error("--namespace is required unless --local-release is used")
        try:
            from modelscope import snapshot_download
        except ImportError as error:
            raise RuntimeError("install modelscope before remote download") from error
        repo_dir = Path(snapshot_download(
            f"{args.namespace}/{repository}", local_dir=str(args.output),
            # Repositories contain several large variants. Fetch only the
            # requested immutable package, rather than the entire family.
            allow_patterns=[f"{relative.as_posix()}/**", "README.md"],
            token=os.environ.get("MODELSCOPE_API_TOKEN"),
        ))
        target = repo_dir / relative
    verify(target)
    print(target.resolve())


if __name__ == "__main__":
    main()
