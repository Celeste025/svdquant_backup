"""Download pinned FastH3 V2 BF16/PTQ/QAD DiT assets without shared components."""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download
from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
import requests


ASSETS = {
    "bf16": (
        "FastVideo/FastVideo-FastH3-Comfy",
        ["diffusion_models/fastvideo_fasth3_8step_v2_pruned_bf16.safetensors"],
    ),
    "ptq": (
        "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Comfy",
        ["diffusion_models/fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors"],
    ),
    "qad": (
        "FastVideo/FastH3-QAD-v2",
        [
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "transformer/adaln_tables.pt",
            *[
                f"transformer/diffusion_pytorch_model-{part:05d}-of-00006.safetensors"
                for part in range(1, 7)
            ],
            "transformer/nvfp4_weights.safetensors",
        ],
    ),
    "qad_core": (
        "FastVideo/FastH3-QAD-v2",
        [
            "transformer/config.json",
            "transformer/nvfp4_weights.safetensors",
        ],
    ),
    "qad_runtime": (
        "FastVideo/FastH3-QAD-v2",
        [
            "configuration.json",
            "model_index.json",
            "modular_model_index.json",
            "scheduler/scheduler_config.json",
            "audio_scheduler/scheduler_config.json",
            "processor/chat_template.json",
            "processor/merges.txt",
            "processor/preprocessor_config.json",
            "processor/tokenizer.json",
            "processor/tokenizer_config.json",
            "processor/video_preprocessor_config.json",
            "processor/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "audio_vae/config.json",
            "audio_vae/diffusion_pytorch_model.safetensors",
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asset", choices=ASSETS, action="append", required=True)
    parser.add_argument("--backend", choices=("huggingface", "modelscope"), default="huggingface")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    hf_api = HfApi() if args.backend == "huggingface" else None
    ms_api = HubApi() if args.backend == "modelscope" else None
    for asset in args.asset:
        repo_id, filenames = ASSETS[asset]
        if args.backend == "modelscope":
            revision = "master"
            inventory = {
                file["Path"]: {"size": file["Size"]}
                for file in ms_api.get_model_files(repo_id, recursive=True)
                if "Size" in file
            }
            # ModelScope's linked ETag is the original LFS SHA256 (verified against
            # the already downloaded AdaLN table), so no HF API is needed here.
            for filename in filenames:
                response = requests.head(
                    f"https://modelscope.cn/models/{repo_id}/resolve/master/{filename}",
                    allow_redirects=True,
                    timeout=(30, 120),
                )
                response.raise_for_status()
                sha256 = response.headers.get("X-Linked-Etag", "").strip('"')
                if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                    raise RuntimeError(f"ModelScope did not supply a SHA256 ETag for {filename}")
                inventory[filename]["sha256"] = sha256
        else:
            revision = hf_api.model_info(repo_id).sha
            inventory = {
                file.path: {
                    "size": file.size,
                    "sha256": file.lfs.sha256 if file.lfs is not None else None,
                }
                for file in hf_api.list_repo_tree(repo_id, recursive=True, revision=revision)
                if hasattr(file, "size")
            }
        destination = args.root / ("qad" if asset in {"qad_core", "qad_runtime"} else asset)
        destination.mkdir(exist_ok=True)
        for filename in filenames:
            source = inventory[filename]
            expected = source["size"]
            print(f"{datetime.now(timezone.utc).isoformat()} {asset} {filename} {expected}", flush=True)
            for attempt in range(1, 13):
                try:
                    if args.backend == "modelscope":
                        path = Path(
                            model_file_download(
                                model_id=repo_id,
                                file_path=filename,
                                local_dir=str(destination),
                            )
                        )
                    else:
                        path = Path(
                            hf_hub_download(
                                repo_id=repo_id,
                                filename=filename,
                                revision=revision,
                                local_dir=destination,
                            )
                        )
                    break
                except Exception as exc:
                    print(f"RETRY {asset} {filename} {attempt}/12: {exc}", flush=True)
                    if attempt == 12:
                        raise
                    time.sleep(min(10 * attempt, 60))
            actual = path.stat().st_size
            if actual != expected:
                raise RuntimeError(f"Size mismatch: {path}: {actual} != {expected}")
            if source["sha256"] is not None:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != source["sha256"]:
                    raise RuntimeError(f"SHA256 mismatch: {path}: {digest} != {source['sha256']}")
        manifest = {
            "asset": asset,
            "repo_id": repo_id,
            "revision": revision,
            "backend": args.backend,
            "files": {
                filename: {
                    "size": inventory[filename]["size"],
                    "sha256": inventory[filename]["sha256"],
                }
                for filename in filenames
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_name = f"download_manifest_{asset}.json" if asset.startswith("qad_") else "download_manifest.json"
        (destination / manifest_name).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"COMPLETE {asset} {revision}", flush=True)


if __name__ == "__main__":
    main()
