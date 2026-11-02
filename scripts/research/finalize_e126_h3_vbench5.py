#!/usr/bin/env python3
"""Wait for E126 GPU queues, then validate/score/render without model polling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/research/E126"
STATUS = REPORT / "finalize_status.json"
LOGS = ROOT / "results/logs"
ARMS = ("bf16", "sage3_fixed", "split_mixed")
GPUS = (5, 6, 7)
DEADLINE_SECONDS = 3 * 3600


def save_status(**kwargs) -> None:
    value = {"format": "e126-finalize-status-v1", "unix_time": time.time(), **kwargs}
    temporary = STATUS.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(STATUS)


def session_active(gpu: int) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"e126_gpu{gpu}"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def main() -> None:
    manifest = json.loads((REPORT / "manifest.json").read_text())
    cases = [row["id"] for row in manifest["prompts"]]
    expected = [REPORT / case / f"{arm}_video.json" for case in cases for arm in ARMS]
    started = time.monotonic()
    try:
        while True:
            done = [path for path in expected if path.is_file()]
            completed_gpus = [gpu for gpu in GPUS if
                              f"GROUP_COMPLETE gpu={gpu}" in
                              (LOGS / f"e126_h3_vbench5_gpu{gpu}.log").read_text()]
            save_status(status="waiting", videos_complete=len(done),
                        videos_expected=len(expected), groups_complete=completed_gpus)
            if len(done) == len(expected) and len(completed_gpus) == len(GPUS):
                break
            if any(not session_active(gpu) and gpu not in completed_gpus for gpu in GPUS):
                raise RuntimeError("An E126 GPU session exited without GROUP_COMPLETE; inspect logs")
            if time.monotonic() - started > DEADLINE_SECONDS:
                raise TimeoutError("E126 exceeded three-hour finalizer watchdog")
            time.sleep(120)
        save_status(status="scoring", videos_complete=len(expected),
                    videos_expected=len(expected), groups_complete=list(GPUS))
        subprocess.run([sys.executable,
                        str(ROOT / "scripts/research/summarize_e126_h3_vbench5_attention.py"),
                        "--gpu", "6"], cwd=ROOT, check=True)
        subprocess.run([sys.executable,
                        str(ROOT / "scripts/research/render_e126_triplet_previews.py")],
                       cwd=ROOT, check=True)
        previews = list(Path("/data1/models/svdquant-wjq/research/20261023/"
                             "E126_h3_vbench5_attention/previews").glob("*.mp4"))
        if len(previews) != 5 or not (REPORT / "summary.json").is_file():
            raise RuntimeError("E126 metrics/previews incomplete")
        save_status(status="complete", videos_complete=len(expected),
                    videos_expected=len(expected), groups_complete=list(GPUS),
                    previews_complete=len(previews), summary=str(REPORT / "summary.json"))
        print("E126 FINALIZE COMPLETE", flush=True)
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
