"""Assemble a local FastVideo QAD-v2 inference root without duplicate weights.

All model files are symlinks. The dense transformer index points at range-
downloaded shards with the 100 AdaLN projections omitted. The official
precomputed AdaLN table must be enabled at inference with
FASTVIDEO_H3_ADALN_TABLE=<root>/transformer/adaln_tables.pt.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def link(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"Existing symlink points elsewhere: {target}")
        return
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qad = args.assets / "qad"
    slim = args.assets / "qad_slim" / "transformer"
    index = json.loads((slim / "diffusion_pytorch_model.safetensors.index.json").read_text())
    if any("adaln_proj.linear" in key and key.startswith("transformer_blocks.")
           for key in index["weight_map"]):
        raise ValueError("Slim index still contains per-block AdaLN projections")
    for filename in set(index["weight_map"].values()):
        path = slim / filename
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            expected = {key for key, file in index["weight_map"].items() if file == filename}
            if set(opened.keys()) != expected:
                raise ValueError(f"Index and tensor keys disagree: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("fastvideo_inference.json", "configuration.json", "model_index.json",
                 "modular_model_index.json"):
        link(qad / name, args.output / name)
    for name in ("processor", "tokenizer", "vae", "audio_vae", "scheduler", "audio_scheduler"):
        link(qad / name, args.output / name)
    link(args.shared / "text_encoder", args.output / "text_encoder")
    transformer = args.output / "transformer"
    transformer.mkdir(exist_ok=True)
    for name in ("config.json", "nvfp4_weights.safetensors", "adaln_tables.pt"):
        link(qad / "transformer" / name, transformer / name)
    for name in set(index["weight_map"].values()) | {"diffusion_pytorch_model.safetensors.index.json"}:
        link(slim / name, transformer / name)
    print(f"Prepared local AdaLN-slim QAD root: {args.output}")
    print(f"Kept dense tensors: {len(index['weight_map'])}; shards: {len(set(index['weight_map'].values()))}")


if __name__ == "__main__":
    main()
