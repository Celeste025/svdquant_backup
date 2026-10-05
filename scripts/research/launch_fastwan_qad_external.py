#!/usr/bin/env python3
"""Bounded single-GPU launch of the official FastWan product reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from resume_h3_plain_baseline import idle_stable

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / "results/research/E024"
PYTHON = "/data1/models/svdquant-wjq/research/envs/fastwan-qad-20261003/bin/python"
RUNNER = ROOT / "scripts/research/run_fastwan_qad_external.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "suite"], required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--deadline-unix", type=float)
    args = parser.parse_args()
    assert args.attempt >= 1
    suffix = "" if args.attempt == 1 else f"_attempt{args.attempt}"
    result = RD / f"{args.phase}_run{suffix}.json"
    output = RD / f"{args.phase}_launcher{suffix}.json"
    logfile = ROOT / "results/logs" / f"E024_{args.phase}{suffix}.log"
    assert not any(p.exists() for p in (result, output, logfile)), "Preserve existing attempt"
    start = time.time()
    deadline = args.deadline_unix or start + 1200
    assert deadline > start
    command = [PYTHON, "-u", str(RUNNER), "--phase", args.phase,
               "--output", str(result), "--deadline-unix", str(deadline)]
    artifacts = Path("/data1/models/svdquant-wjq/research/20261003/E024/videos") / (args.phase + suffix)
    command += ["--artifact-dir", str(artifacts)]
    record = dict(experiment="E024", status="running", phase=args.phase,
                  attempt=args.attempt, gpu=0, start_epoch=start,
                  deadline_epoch=deadline, command=command, log=str(logfile),
                  source_sha256=sha(RUNNER), launcher_sha256=sha(__file__))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_HOME="/usr/local/cuda",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               FASTVIDEO_ATTENTION_BACKEND="ATTN_QAT_INFER", FASTVIDEO_STAGE_LOGGING="1",
               OMP_NUM_THREADS="6", MAX_JOBS="4", PYTHONUNBUFFERED="1")
    cache = Path("/data1/models/svdquant-wjq/research/20261003/E024/environment/cache")
    for name, subdir in {
        "XDG_CACHE_HOME": ".", "HF_HOME": "huggingface", "TORCH_HOME": "torch",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "TMPDIR": "tmp",
    }.items():
        path = cache / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache.parent / "flashinfer_workspace")
    env["PATH"] = str(Path(PYTHON).parent) + ":/usr/local/cuda/bin:" + env.get("PATH", "")
    proc = None
    try:
        record["gpu_before"] = idle_stable(0)
        with logfile.open("x") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = proc.pid
            write(output, record)
            try:
                code = proc.wait(timeout=max(0.01, deadline - time.time()))
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
        record["returncode"] = code
        if code:
            raise RuntimeError(f"Official product worker exited {code}; see retained log")
        value = json.loads(result.read_text())
        assert value["status"] == "complete", "Worker did not complete"
        record.update(status="complete", result=str(result), result_sha256=sha(result))
    except BaseException:
        record.update(status="failed_stop", error=traceback.format_exc())
        raise
    finally:
        record["seconds"] = time.time() - start
        write(output, record)
        print(record["status"], record["seconds"], flush=True)


if __name__ == "__main__":
    main()
