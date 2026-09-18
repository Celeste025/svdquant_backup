#!/usr/bin/env python3
"""Build immutable, self-contained SVDQuant ModelScope staging packages.

This never uploads.  It copies only an explicit allowlist of final artifacts
and writes checksums after every copy has completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import ssl
from urllib.request import urlopen
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data1/models/svdquant-wjq")
CKPTS = DATA / "ckpts"
RUNS = DATA / "runs"
THIRD_PARTY_CONFIG = ROOT / "third_party/deepcompressor/examples/diffusion/configs"


@dataclass(frozen=True)
class Spec:
    family: str
    repository: str
    variant: str
    source: Path
    required: tuple[str, ...]
    base_model: str
    recipe: dict
    configs: tuple[Path, ...] = ()
    license_source: Path | None = None
    license_url: str = ""
    license_notice: str = ""
    cache_tag: str | None = None


SPECS = (
    Spec("rcm-wan", "svdquant-rcm-wan2.1-1.3b", "real-nvfp4-g5-r32",
         CKPTS / "rcm-wan2.1-1.3b-real-nvfp4-s16", ("model.pt", "scale.pt", "wgts.pt"),
         "Wan-AI/Wan2.1-T2V-1.3B + local rCM transformer", {"format": "real-NVFP4 W4A4", "rank": 32, "grid": 5, "group_size": 16},
         (THIRD_PARTY_CONFIG / "svdquant/real_nvfp4.yaml", THIRD_PARTY_CONFIG / "svdquant/wan_s16.yaml"),
         license_source=DATA / "models/Wan2.1-T2V-1.3B/LICENSE.txt"),
    Spec("rcm-wan", "svdquant-rcm-wan2.1-1.3b", "int4-g10-r32",
         CKPTS / "rcm-wan2.1-1.3b-int4-s16-g10", ("model.pt", "scale.pt", "wgts.pt", "smooth.pt", "branch.pt"),
         "Wan-AI/Wan2.1-T2V-1.3B + local rCM transformer", {"format": "dynamic signed INT4 W4A4", "rank": 32, "grid": 10, "group_size": 64},
         (THIRD_PARTY_CONFIG / "svdquant/int4.yaml", THIRD_PARTY_CONFIG / "svdquant/rcm_wan_int4_s16_g10.yaml"),
         license_source=DATA / "models/Wan2.1-T2V-1.3B/LICENSE.txt",
         cache_tag="rcm-wan2.1-1.3b-int4-s16-g10"),
    Spec("rcm-wan", "svdquant-rcm-wan2.1-1.3b", "real-nvfp4-g20-r32",
         CKPTS / "rcm-wan2.1-1.3b-real-nvfp4-s16-g20-r32", ("model.pt", "scale.pt", "wgts.pt", "manifest.json"),
         "Wan-AI/Wan2.1-T2V-1.3B + local rCM transformer", {"format": "real-NVFP4 W4A4", "rank": 32, "grid": 20, "group_size": 16},
         (THIRD_PARTY_CONFIG / "svdquant/real_nvfp4.yaml", THIRD_PARTY_CONFIG / "svdquant/wan_s16.yaml", THIRD_PARTY_CONFIG / "svdquant/rcm_wan_real_nvfp4_s16_g20_r32.yaml"),
         license_source=DATA / "models/Wan2.1-T2V-1.3B/LICENSE.txt"),
    Spec("rcm-wan", "svdquant-rcm-wan2.1-1.3b", "real-nvfp4-g10-r64",
         CKPTS / "rcm-wan2.1-1.3b-real-nvfp4-s16-g10-r64", ("model.pt", "scale.pt", "wgts.pt", "manifest.json"),
         "Wan-AI/Wan2.1-T2V-1.3B + local rCM transformer", {"format": "real-NVFP4 W4A4", "rank": 64, "grid": 10, "group_size": 16},
         (THIRD_PARTY_CONFIG / "svdquant/real_nvfp4.yaml", THIRD_PARTY_CONFIG / "svdquant/wan_s16.yaml", THIRD_PARTY_CONFIG / "svdquant/rcm_wan_real_nvfp4_s16_g10_r64.yaml"),
         license_source=DATA / "models/Wan2.1-T2V-1.3B/LICENSE.txt"),
    Spec("minimax-h3", "svdquant-minimax-h3", "nvfp4-g10-r32",
         ROOT / "results/checkpoints/minimax_h3_svdquant_standard_8p64s", ("quant_state.pt", "summary.json"),
         "MiniMaxAI/MiniMax-H3 FL2VA", {"format": "real-NVFP4 W4A4", "rank": 32, "grid": 10, "group_size": 16},
         (ROOT / "configs/minimax_h3_svdquant_standard_8p64s.json",),
         license_notice="Upstream MiniMax H3 Community License: https://huggingface.co/MiniMaxAI/MiniMax-H3",
         license_url="https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/LICENSE"),
    Spec("minimax-h3", "svdquant-minimax-h3", "nvfp4-g10-r64",
         ROOT / "results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s", ("quant_state.pt", "summary.json"),
         "MiniMaxAI/MiniMax-H3 FL2VA", {"format": "real-NVFP4 W4A4", "rank": 64, "grid": 10, "group_size": 16},
         (ROOT / "configs/minimax_h3_svdquant_standard_8p64s.json",),
         license_notice="Upstream MiniMax H3 Community License: https://huggingface.co/MiniMaxAI/MiniMax-H3",
         license_url="https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/LICENSE"),
    Spec("flux1", "svdquant-flux1", "dev-int4-r32",
         CKPTS / "flux.1-dev-int4-fast-s64-lowmem", ("model.pt", "scale.pt", "wgts.pt", "smooth.pt", "branch.pt"),
         "black-forest-labs/FLUX.1-dev", {"format": "signed INT4 W4A4", "rank": 32, "grid": 10, "group_size": 64},
         (THIRD_PARTY_CONFIG / "model/flux.1-dev.yaml", THIRD_PARTY_CONFIG / "svdquant/int4.yaml", THIRD_PARTY_CONFIG / "svdquant/fast.yaml", THIRD_PARTY_CONFIG / "svdquant/lowmem.yaml"),
         license_source=DATA / "models/FLUX.1-dev/LICENSE.md", cache_tag="flux_fast_s64"),
    Spec("flux1", "svdquant-flux1", "schnell-int4-r32",
         CKPTS / "flux.1-schnell-int4-fast-s64-lowmem", ("model.pt", "scale.pt", "wgts.pt", "smooth.pt", "branch.pt"),
         "black-forest-labs/FLUX.1-schnell", {"format": "signed INT4 W4A4", "rank": 32, "grid": 10, "group_size": 64},
         (THIRD_PARTY_CONFIG / "model/flux.1-schnell.yaml", THIRD_PARTY_CONFIG / "svdquant/int4.yaml", THIRD_PARTY_CONFIG / "svdquant/fast.yaml", THIRD_PARTY_CONFIG / "svdquant/lowmem.yaml"),
         license_notice="Upstream FLUX.1-schnell is Apache-2.0: https://huggingface.co/black-forest-labs/FLUX.1-schnell",
         license_url="https://raw.githubusercontent.com/black-forest-labs/flux/main/LICENSE", cache_tag="flux_schnell_fast_s64"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_state(spec: Spec, kind: str) -> Path:
    assert spec.cache_tag is not None
    root = RUNS / spec.cache_tag / "diffusion/cache"
    candidates = [path for path in root.rglob("*.pt") if f"/{kind}/" in str(path)]
    if len(candidates) != 1:
        raise RuntimeError(f"{spec.variant}: expected one {kind} state below {root}, found {len(candidates)}")
    return candidates[0]


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_package(spec: Spec, release_root: Path, revision: str, overwrite: bool) -> Path:
    destination = release_root / spec.repository / "variants" / spec.variant
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite staged package {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    sources = {name: spec.source / name for name in spec.required if name not in {"smooth.pt", "branch.pt"}}
    if "smooth.pt" in spec.required:
        sources["smooth.pt"] = cache_state(spec, "smooth")
        sources["branch.pt"] = cache_state(spec, "branch")
    for name, source in sources.items():
        copy(source, destination / name)
    for config in spec.configs:
        copy(config, destination / "recipe" / config.name)
    if spec.license_source is not None:
        copy(spec.license_source, destination / "LICENSE-UPSTREAM.md")
    elif spec.license_url:
        # The PTQ environment has an incomplete system CA store; certifi keeps
        # this reproducible without disabling certificate verification.
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urlopen(spec.license_url, timeout=60, context=context) as response:
            (destination / "LICENSE-UPSTREAM.md").write_bytes(response.read())
        # The H3 license requires this exact notice next to any redistribution.
        if spec.family == "minimax-h3":
            (destination / "NOTICE").write_text(
                "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
                "Copyright © 2026 MiniMax. All Rights Reserved.\n", encoding="utf-8")
    else:
        (destination / "LICENSE-NOTICE.md").write_text(spec.license_notice + "\n", encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "artifact_kind": "svdquant-quantized-state",
        "family": spec.family,
        "variant": spec.variant,
        "base_model": spec.base_model,
        "recipe": spec.recipe,
        "required_files": list(spec.required),
        "loader": {"repository_commit": revision, "entrypoint": "svdquant-exp registry"},
        "source_paths_removed": True,
    }
    (destination / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    readme = (
        f"# {spec.family}: {spec.variant}\n\n"
        "This is a SVDQuant quantized-state artifact, not a standalone base model. "
        f"Download `{spec.base_model}` separately, then use the matching svdquant-exp loader.\n\n"
        f"Required files: {', '.join(spec.required)}.\n"
    )
    (destination / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(path for path in destination.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (destination / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(destination)}\n" for path in files))
    return destination


def write_repository_card(release_root: Path, repository: str) -> None:
    variants = [spec for spec in SPECS if spec.repository == repository]
    card = [f"# {repository}", "", "SVDQuant quantized-state release. Download the upstream base model separately; each variant has its own README, artifact manifest, recipe snapshot, license material, and SHA256SUMS.", "", "## Variants", ""]
    for spec in variants:
        recipe = spec.recipe
        card.append(f"- `variants/{spec.variant}` — {recipe['format']}; rank={recipe['rank']}, grid={recipe['grid']}.")
    (release_root / repository / "README.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DATA / "releases/modelscope/v0.1.0")
    parser.add_argument("--family", action="append", choices=("rcm-wan", "minimax-h3", "flux1"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cards-only", action="store_true", help="write repository-level README files without touching variants")
    args = parser.parse_args()
    revision = git_revision()
    selected = [spec for spec in SPECS if not args.family or spec.family in args.family]
    if not args.cards_only:
        for spec in selected:
            print(write_package(spec, args.release_root, revision, args.overwrite), flush=True)
    for repository in sorted({spec.repository for spec in selected}):
        write_repository_card(args.release_root, repository)


if __name__ == "__main__":
    main()
