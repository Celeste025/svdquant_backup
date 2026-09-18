#!/usr/bin/env python3
"""Verify every SHA256SUMS file in an immutable local ModelScope release."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT = Path("/data1/models/svdquant-wjq/releases/modelscope/v0.1.0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(sums)
    for line in sums.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        candidate = directory / relative.strip()
        if not candidate.is_file() or sha256(candidate) != expected:
            raise RuntimeError(f"checksum mismatch: {candidate}")
    print(f"OK {directory}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    roots = [*sorted(args.release_root.glob("*/variants/*")), *sorted((args.release_root / "datasets").glob("*"))]
    if len(roots) != 10:
        raise RuntimeError(f"expected eight model variants and two datasets, found {len(roots)} packages")
    for directory in roots:
        verify(directory)


if __name__ == "__main__":
    main()
