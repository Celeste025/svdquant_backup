#!/usr/bin/env python3
"""Audit matched E129 rank-64 SVDQuant videos against E124/E126 BF16 arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E124 = ROOT / "results/research/E124"
E126 = ROOT / "results/research/E126"
E129 = ROOT / "results/research/E129"
ARM = "svdquant_bf16_attention"
STATE = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_r64_8p64s/quant_state.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", help="Audit only selected case IDs")
    args = parser.parse_args()
    source = {f"p{int(prompt_id):03d}": (E124, 20, 576, 1024)
              for ids in json.loads((E124 / "manifest.json").read_text())["groups"].values()
              for prompt_id in ids}
    source.update({row["id"]: (E126, 50, 768, 1344)
                   for row in json.loads((E126 / "manifest.json").read_text())["prompts"]})
    assert len(source) == 17
    if args.cases:
        if len(set(args.cases)) != len(args.cases) or not set(args.cases) <= set(source):
            raise RuntimeError("invalid case selection")
        source = {case: source[case] for case in args.cases}
    state_hash = sha256(STATE)
    summary = {"format": "e129-svdquant-final-audit-v1", "expected_videos": len(source),
               "state_sha256": state_hash, "cases": {}, "complete_videos": 0,
               "incomplete": []}
    for case, (baseline, steps, height, width) in source.items():
        reference = json.loads((baseline / case / "bf16_denoise.json").read_text())
        denoise_path = E129 / case / f"{ARM}_denoise.json"
        video_path = E129 / case / f"{ARM}_video.json"
        if not denoise_path.is_file() or not video_path.is_file():
            summary["incomplete"].append({"case": case, "reason": "missing report"})
            continue
        d, v = json.loads(denoise_path.read_text()), json.loads(video_path.read_text())
        media = Path(v["video"])
        checks = {
            "denoise_complete": d.get("status") == "denoise_complete",
            "step_count": len(d.get("steps", [])) == steps,
            "step_finite": all(math.isfinite(row.get("dit_seconds", float("nan"))) and
                               row.get("dit_seconds", 0) > 0 and
                               math.isfinite(row.get("video_output_energy", float("nan"))) and
                               math.isfinite(row.get("audio_output_energy", float("nan")))
                               for row in d.get("steps", [])),
            "initial_latent_pairing": d.get("initial_latent_sha256") ==
                                      reference["initial_latent_sha256"],
            "settings": d.get("settings", {}).get("height") == height and
                        d.get("settings", {}).get("width") == width and
                        d.get("settings", {}).get("steps") == steps and
                        d.get("seed") == reference.get("seed"),
            "state_hash": d.get("linears", {}).get("state_sha256") == state_hash,
            "quantization": d.get("linears", {}).get("count") == 200 and
                            d.get("linears", {}).get("rank") == 64,
            "bf16_attention": d.get("attention_router", {}).get("e129_arm") == ARM,
            "video_shape": v.get("frames") == 124 and v.get("height") == height and
                           v.get("width") == width and v.get("audio_streams") == 1,
            "media_exists": media.is_file(),
            "media_hash": media.is_file() and sha256(media) == v.get("sha256"),
        }
        summary["cases"][case] = {"baseline": str(baseline), "checks": checks,
                                  "video": str(media), "sha256": v.get("sha256"),
                                  "sum_dit_seconds": sum(row["dit_seconds"]
                                                         for row in d.get("steps", []))}
        if all(checks.values()):
            summary["complete_videos"] += 1
        else:
            summary["incomplete"].append({"case": case,
                                          "reason": [k for k, ok in checks.items() if not ok]})
    target = E129 / "final_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": summary["complete_videos"],
                      "expected": summary["expected_videos"],
                      "incomplete": summary["incomplete"], "report": str(target)},
                     ensure_ascii=False, indent=2))
    if summary["incomplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
