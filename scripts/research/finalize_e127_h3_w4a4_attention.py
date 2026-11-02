#!/usr/bin/env python3
"""Audit E127/E128 videos against E124/E126 paired-generation contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E124 = ROOT / "results/research/E124"
E126 = ROOT / "results/research/E126"
ARMS = ("w4a4_bf16_attention", "w4a4_sage3_fixed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E127", "E128"), default="E127")
    args = parser.parse_args()
    report_dir = ROOT / "results/research" / args.experiment
    source = {f"p{int(prompt_id):03d}": (E124, 20, 576, 1024)
              for ids in json.loads((E124 / "manifest.json").read_text())["groups"].values()
              for prompt_id in ids}
    source.update({row["id"]: (E126, 50, 768, 1344)
                   for row in json.loads((E126 / "manifest.json").read_text())["prompts"]})
    assert len(source) == 17
    summary = {"format": f"{args.experiment.lower()}-h3-w4a4-attention-final-audit-v1",
               "expected_videos": 34,
               "cases": {}, "complete_videos": 0, "incomplete": []}
    for case, (baseline, steps, height, width) in source.items():
        reference = json.loads((baseline / case / "bf16_denoise.json").read_text())
        baseline_initial = reference["initial_latent_sha256"]
        summary["cases"][case] = {"baseline": str(baseline), "arms": {}}
        for arm in ARMS:
            denoise_path = report_dir / case / f"{arm}_denoise.json"
            video_path = report_dir / case / f"{arm}_video.json"
            if not denoise_path.is_file() or not video_path.is_file():
                summary["incomplete"].append({"case": case, "arm": arm, "reason": "missing report"})
                continue
            d, v = json.loads(denoise_path.read_text()), json.loads(video_path.read_text())
            media = Path(v["video"])
            checks = {
                "denoise_complete": d.get("status") == "denoise_complete",
                "step_count": len(d.get("steps", [])) == steps,
                "step_finite": all(row.get("dit_seconds", -1) > 0 and
                                   row.get("video_output_energy", -1) >= 0 and
                                   row.get("audio_output_energy", -1) >= 0
                                   for row in d.get("steps", [])),
                "initial_latent_pairing": d.get("initial_latent_sha256") == baseline_initial,
                "settings": d.get("settings", {}).get("height") == height and
                            d.get("settings", {}).get("width") == width and
                            d.get("settings", {}).get("steps") == steps,
                "video_shape": v.get("frames") == 124 and v.get("height") == height and
                               v.get("width") == width and v.get("audio_streams", 1) == 1,
                "media_exists": media.is_file(),
                "media_hash": media.is_file() and sha256(media) == v.get("sha256"),
            }
            summary["cases"][case]["arms"][arm] = {"checks": checks,
                    "video": str(media), "sha256": v.get("sha256"),
                    "sum_dit_seconds": sum(x["dit_seconds"] for x in d.get("steps", []))}
            if all(checks.values()):
                summary["complete_videos"] += 1
            else:
                summary["incomplete"].append({"case": case, "arm": arm,
                                               "reason": [k for k, ok in checks.items() if not ok]})
    target = report_dir / "final_audit.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": summary["complete_videos"],
                      "expected": summary["expected_videos"],
                      "incomplete": summary["incomplete"], "report": str(target)},
                     ensure_ascii=False, indent=2))
    if summary["incomplete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
