#!/usr/bin/env python3
"""Read-only oracle test: subtract a prior paired DiT error from the next one."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"race": ROOT / "results/research/E103/manifest.json",
           "clap0": ROOT / "results/research/E104/manifest.json"}
COEFFICIENTS = {"k05": 0.5, "k08": 0.8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_modality(error: torch.Tensor, previous: torch.Tensor,
                     reference: torch.Tensor) -> dict:
    if error.shape != previous.shape or error.shape != reference.shape:
        raise RuntimeError("unaligned output coordinates or shapes")
    current = error.reshape(-1).to(torch.float64)
    prior = previous.reshape(-1).to(torch.float64)
    signal = reference.reshape(-1).to(torch.float64)
    e2 = float(torch.dot(current, current))
    p2 = float(torch.dot(prior, prior))
    cross = float(torch.dot(current, prior))
    denominator = float(torch.dot(signal, signal))
    if min(e2, p2, denominator) <= 0 or not all(map(math.isfinite, (e2, p2, cross, denominator))):
        raise RuntimeError("nonfinite or zero output energy")
    result = {"error_energy": e2, "previous_error_energy": p2,
              "cross_energy": cross, "reference_energy": denominator,
              "baseline_nmse": e2 / denominator,
              "adjacent_error_cosine": cross / math.sqrt(e2 * p2),
              "previous_over_current_error_rms": math.sqrt(p2 / e2),
              "k_opt_oracle": cross / p2}
    result["best_per_step_oracle_nmse"] = max(0.0, e2 - cross * cross / p2) / denominator
    for name, coefficient in COEFFICIENTS.items():
        corrected = e2 - 2 * coefficient * cross + coefficient * coefficient * p2
        if corrected < -1e-8:
            raise RuntimeError("negative corrected error energy")
        result[f"{name}_error_energy"] = max(0.0, corrected)
        result[f"{name}_nmse"] = max(0.0, corrected) / denominator
        result[f"{name}_relative_nmse_reduction"] = 1 - max(0.0, corrected) / e2
    return result


def main() -> None:
    torch.set_num_threads(6)
    output = ROOT / "results/research/E107"
    json_path = output / "previous_error_subtraction.json"
    csv_path = output / "per_step.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite existing E107 outputs")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"format": "h3-e107-previous-step-oracle-error-subtraction-v1",
               "definition": "e_t = paired SVDQuant output minus BF16 output on the saved BF16 trajectory; corrected=e_t-k*e_(t-1); t=1..19",
               "coefficients": COEFFICIENTS, "cases": {}}
    flat_rows = []
    for case, manifest_path in SOURCES.items():
        manifest = json.loads(manifest_path.read_text())
        records = manifest["records"]
        if len(records) != 20 or [item["step"] for item in records] != list(range(20)):
            raise RuntimeError(f"incomplete saved calls: {case}")
        case_result = {"prompt_id": manifest["prompt_id"], "seed": manifest["seed"],
                       "source_manifest": str(manifest_path.resolve()),
                       "source_manifest_sha256": sha256(manifest_path),
                       "per_step": [], "summary": {}}
        prior = None
        for record in records:
            path = Path(record["path"])
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"raw capture checksum mismatch: {path}")
            data = torch.load(path, map_location="cpu", weights_only=False)
            step = record["step"]
            if data["step"] != step:
                raise RuntimeError(f"raw step mismatch: {case}/{step}")
            current = {}
            for modality in ("video", "audio"):
                reference = data[f"{modality}_bf16"].float()
                error = data[f"{modality}_error_fp32"].float()
                quantized = data[f"{modality}_svdquant"].float()
                if not torch.equal(error, quantized - reference):
                    raise RuntimeError(f"raw error differs from paired output difference: {case}/{step}/{modality}")
                current[modality] = error
                if prior is None:
                    continue
                row = analyze_modality(error, prior[modality], reference)
                if abs(row["baseline_nmse"] - record["metrics"][modality]["nmse"]) > 1e-7:
                    raise RuntimeError(f"paired baseline NMSE mismatch: {case}/{step}/{modality}")
                row.update(case=case, modality=modality, step=step)
                case_result["per_step"].append(row)
                flat_rows.append(row)
            prior = current
        for modality in ("video", "audio"):
            rows = [row for row in case_result["per_step"] if row["modality"] == modality]
            reference_total = sum(row["reference_energy"] for row in rows)
            error_total = sum(row["error_energy"] for row in rows)
            modality_summary = {"steps": len(rows), "baseline_pooled_nmse": error_total / reference_total,
                                "mean_adjacent_error_cosine": sum(row["adjacent_error_cosine"] for row in rows) / len(rows),
                                "median_k_opt_oracle": float(torch.tensor([row["k_opt_oracle"] for row in rows]).median())}
            for name in COEFFICIENTS:
                corrected_total = sum(row[f"{name}_error_energy"] for row in rows)
                modality_summary[f"{name}_pooled_nmse"] = corrected_total / reference_total
                modality_summary[f"{name}_pooled_relative_reduction"] = 1 - corrected_total / error_total
                modality_summary[f"{name}_steps_improved"] = sum(
                    row[f"{name}_relative_nmse_reduction"] > 0 for row in rows)
            case_result["summary"][modality] = modality_summary
        payload["cases"][case] = case_result
    with json_path.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    columns = ["case", "modality", "step", "baseline_nmse", "k05_nmse", "k05_relative_nmse_reduction",
               "k08_nmse", "k08_relative_nmse_reduction", "adjacent_error_cosine",
               "previous_over_current_error_rms", "k_opt_oracle", "best_per_step_oracle_nmse"]
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({case: item["summary"] for case, item in payload["cases"].items()}, indent=2))
    print(f"COMPLETE {json_path} {csv_path}")


if __name__ == "__main__":
    main()
