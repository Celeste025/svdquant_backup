"""Range-download QAD-v2 dense shards, omitting AdaLN projections covered by tables.

The packed NVFP4 file and the 8-step AdaLN table are separate assets. This
script rewrites only the six dense safetensors shards and their index. It does
not modify weights, dequantize FP4, or approximate the timestep tables.
"""

import argparse
import hashlib
import json
import re
import struct
import time
from pathlib import Path

import requests
from safetensors import safe_open


REPO = "FastVideo/FastH3-QAD-v2"
ADALN = re.compile(r"^transformer_blocks\.\d+\.adaln_proj\.linear\.(weight|bias)$")


def get_range(session: requests.Session, url: str, start: int, end: int, writer) -> None:
    """Append one exact byte range, retrying from the last byte written."""
    position = start
    failures = 0
    while position <= end:
        try:
            with session.get(
                url,
                headers={"Range": f"bytes={position}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                expected_prefix = f"bytes {position}-{end}/"
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(expected_prefix):
                    raise ValueError(f"Range response mismatch: {response.status_code} {content_range}")
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        writer.write(chunk)
                        position += len(chunk)
                        if position > end + 1:
                            raise ValueError("Server returned more bytes than requested")
            if position <= end:
                raise ConnectionError(f"Range ended early at {position} of {end + 1}")
        except (requests.RequestException, ConnectionError, TimeoutError) as exc:
            failures += 1
            if failures > 30:
                raise RuntimeError(f"Range {start}-{end} failed at byte {position}") from exc
            print(f"RETRY range {start}-{end} at {position}: {exc}", flush=True)
            time.sleep(min(failures * 2, 30))


def read_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    from io import BytesIO

    sink = BytesIO()
    get_range(session, url, start, end, sink)
    return sink.getvalue()


def repack_shard(session: requests.Session, source_name: str, target: Path,
                 expected_names: set[str]) -> dict:
    url = f"https://modelscope.cn/models/{REPO}/resolve/master/transformer/{source_name}"
    header_length = struct.unpack("<Q", read_range(session, url, 0, 7))[0]
    if header_length > 10_000_000:
        raise ValueError(f"Unreasonable safetensors header length: {header_length}")
    header = json.loads(read_range(session, url, 8, 7 + header_length))
    original_metadata = header.pop("__metadata__", None)
    included = {
        name: entry for name, entry in header.items() if not ADALN.fullmatch(name)
    }
    if set(included) != expected_names:
        raise ValueError(f"Index/header disagreement for {source_name}: "
                         f"missing={sorted(expected_names - set(included))[:4]}, "
                         f"extra={sorted(set(included) - expected_names)[:4]}")
    ordered = sorted(included.items(), key=lambda item: item[1]["data_offsets"][0])
    output_header = {}
    offset = 0
    for name, entry in ordered:
        begin, finish = entry["data_offsets"]
        size = finish - begin
        output_header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if original_metadata is not None:
        output_header["__metadata__"] = original_metadata
    encoded = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".incomplete")
    prefix = struct.pack("<Q", len(encoded)) + encoded
    expected_size = len(prefix) + offset
    if partial.exists():
        with partial.open("rb") as reader:
            if reader.read(len(prefix)) != prefix:
                raise ValueError(f"Existing partial has a mismatched header: {partial}")
        if partial.stat().st_size > expected_size:
            raise ValueError(f"Existing partial exceeds expected size: {partial}")
        print(f"RESUME {source_name} at {partial.stat().st_size}/{expected_size} bytes", flush=True)
    else:
        with partial.open("wb") as writer:
            writer.write(prefix)
    with partial.open("ab") as writer:
        output_offset = len(prefix)
        for number, (name, entry) in enumerate(ordered, 1):
            begin, finish = entry["data_offsets"]
            tensor_size = finish - begin
            already_present = max(0, min(tensor_size, writer.tell() - output_offset))
            if already_present < tensor_size:
                get_range(session, url, 8 + header_length + begin + already_present,
                          8 + header_length + finish - 1, writer)
            output_offset += tensor_size
            if number % 20 == 0 or number == len(ordered):
                print(f"{source_name}: {number}/{len(ordered)} tensors", flush=True)
    if partial.stat().st_size != expected_size:
        raise ValueError(f"Output size mismatch for {partial}")
    with safe_open(str(partial), framework="pt", device="cpu") as opened:
        if set(opened.keys()) != expected_names:
            raise ValueError(f"Safetensors key validation failed for {partial}")
    partial.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "file": source_name,
        "kept_tensors": len(included),
        "removed_adaln_tensors": len(header) - len(included),
        "bytes": target.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shards", type=int, default=6)
    args = parser.parse_args()
    source = json.loads(args.index.read_text())
    weight_map = source["weight_map"]
    removed = [name for name in weight_map if ADALN.fullmatch(name)]
    if len(removed) != 100:
        raise ValueError(f"Expected exactly 100 block AdaLN weight/bias tensors, found {len(removed)}")
    retained = {name: shard for name, shard in weight_map.items() if name not in removed}
    shards = sorted(set(weight_map.values()))[:args.max_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with requests.Session() as session:
        for shard in shards:
            target = args.output / shard
            expected_names = {name for name, filename in retained.items() if filename == shard}
            if target.is_file():
                with safe_open(str(target), framework="pt", device="cpu") as opened:
                    if set(opened.keys()) == expected_names:
                        print(f"SKIP validated {target}", flush=True)
                        records.append({"file": shard, "bytes": target.stat().st_size, "existing": True})
                        continue
            records.append(repack_shard(session, shard, target, expected_names))
    slim_index = {
        "metadata": {**source.get("metadata", {}), "total_size": sum(r["bytes"] for r in records)},
        "weight_map": {name: shard for name, shard in retained.items() if shard in shards},
    }
    (args.output / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(slim_index, indent=2) + "\n"
    )
    (args.output / "adaln_prune_manifest.json").write_text(
        json.dumps({"repo": REPO, "removed_adaln_tensors": removed,
                    "shards": records}, indent=2) + "\n"
    )
    print(f"COMPLETE {len(shards)} shards, removed {len(removed)} AdaLN tensors", flush=True)


if __name__ == "__main__":
    main()
