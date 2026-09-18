#!/usr/bin/env python3
"""Download and SHA-256-verify a published SVDQuant video dataset."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


DATASETS = {
    "rcm-wan": "svdquant-videoeval-rcm-wan",
    "minimax-h3": "svdquant-videoeval-minimax-h3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> None:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"missing {sums}")
    for line in sums.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = root / relative.strip()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if sha256(candidate) != expected:
            raise RuntimeError(f"checksum mismatch: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=tuple(DATASETS))
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise RuntimeError("install modelscope before remote download") from error
    repository = DATASETS[args.family]
    target = Path(snapshot_download(
        f"{args.namespace}/{repository}", repo_type="dataset",
        local_dir=str(args.output / repository),
        token=os.environ.get("MODELSCOPE_API_TOKEN"),
    ))
    verify(target)
    print(target.resolve())


if __name__ == "__main__":
    main()
