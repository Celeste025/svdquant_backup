#!/usr/bin/env python3
"""E105: closed-loop H3 paired-DiT output interventions and separate VAE decode."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_minimax_h3_paired_step_error import apply_svdquant, split_output
from minimax_h3_svdquant_common import (
    MiniMaxH3Pipeline, ModelConfig, disk_config, load_h3_pipeline,
    local_h3_files, read_prompt, tree_device,
)

CASES = {"race": (16, 20026), "clap0": (161, 38001610)}
ARMS = ("tail_restore", "rademacher_flat", "gaussian")
STATE = ROOT / "results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt"
E101 = ROOT / "results/research/E101"
DATA = Path("/data1/models/svdquant-wjq/research/20261016/E105_h3_output_interventions")
REPORT = ROOT / "results/research/E105"
PLAN = ROOT / "research_state/06_experiments/decision_value_plan_E105_h3_output_error_interventions.md"
DENOISE_FORMAT = "h3-e105-output-intervention-denoise-v1"
VIDEO_FORMAT = "h3-e105-output-intervention-video-v1"


class DenoiseComplete(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def case_info(case: str) -> dict:
    prompt_id, seed = CASES[case]
    e101 = json.loads((E101 / case / "svdquant.json").read_text())
    require((e101["prompt_id"], e101["seed"], e101["height"], e101["width"],
             e101["frames"], e101["steps"]) == (prompt_id, seed, 576, 1024, 124, 20),
            "E101 case identity/settings mismatch")
    if case == "race":
        row = read_prompt(prompt_id)
        require(row["prompt"] == e101["prompt"] and row["seed"] == seed,
                "race prompt table does not reproduce E101")
    else:
        row = {"prompt": "A person is clapping", "seed": seed}
        require(row["prompt"] == e101["prompt"], "clap prompt differs from E101")
    return {"prompt_id": prompt_id, "seed": seed, "prompt": row["prompt"], "e101": e101}


def estimate(pred: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    diff = pred.float() - reference.float()
    mse = float(diff.square().mean(dtype=torch.float64))
    energy = float(reference.float().square().mean(dtype=torch.float64))
    return mse, mse / max(energy, 1e-30)


def alter(reference: torch.Tensor, quantized: torch.Tensor, arm: str,
          seed: int, step: int, modality_id: int) -> tuple[torch.Tensor, dict]:
    require(reference.shape == quantized.shape and reference.dtype == quantized.dtype == torch.bfloat16,
            "paired outputs must have matching BF16 shape and dtype")
    ref, q = reference.float(), quantized.float()
    error = q - ref
    target_mse = float(error.square().mean(dtype=torch.float64))
    ref_energy = float(ref.square().mean(dtype=torch.float64))
    require(target_mse > 0 and math.isfinite(target_mse) and ref_energy > 0 and
            bool(torch.isfinite(error).all()), "invalid paired output energy")
    row = {"target_mse": target_mse, "target_nmse": target_mse / ref_energy,
           "reference_mean_square": ref_energy, "output_elements": error.numel()}
    if arm == "tail_restore":
        mean = error.mean()
        std = error.std(unbiased=False)
        mask = (error - mean).abs() > (4.0 * std)
        modified = quantized.clone()
        modified[mask] = reference[mask]
        row.update(error_mean=float(mean), error_std=float(std),
                   repaired_elements=int(mask.sum()), repaired_fraction=float(mask.float().mean()),
                   removed_error_energy_share=float(error[mask].square().sum(dtype=torch.float64) /
                                                    error.square().sum(dtype=torch.float64)))
    else:
        generator = torch.Generator(device=reference.device)
        random_seed = seed + (1 if arm == "rademacher_flat" else 2) * 10_000_000 + step * 1000 + modality_id
        generator.manual_seed(random_seed)
        if arm == "rademacher_flat":
            signs = torch.randint(0, 2, reference.shape, device=reference.device,
                                  generator=generator, dtype=torch.int8)
            direction = signs.float().mul_(2).sub_(1)
        elif arm == "gaussian":
            direction = torch.randn(reference.shape, device=reference.device,
                                    dtype=torch.float32, generator=generator)
            direction /= direction.square().mean().sqrt()
        else:
            raise ValueError(arm)
        amplitude = math.sqrt(target_mse)
        low, high = 0.0, 2.0 * amplitude
        for _ in range(8):
            high_candidate = (ref + direction * high).to(torch.bfloat16)
            high_mse, _ = estimate(high_candidate, reference)
            if high_mse >= target_mse:
                break
            high *= 2.0
        else:
            raise RuntimeError("cannot bracket BF16-rounded injected MSE")
        best_candidate, best_amplitude, best_gap = None, None, float("inf")
        for _ in range(18):
            middle = (low + high) / 2.0
            candidate = (ref + direction * middle).to(torch.bfloat16)
            candidate_mse, _ = estimate(candidate, reference)
            gap = abs(candidate_mse / target_mse - 1)
            if gap < best_gap:
                best_candidate, best_amplitude, best_gap = candidate, middle, gap
            if gap <= 0.005:
                break
            if candidate_mse < target_mse:
                low = middle
            else:
                high = middle
        if best_gap > 0.005:
            # A constant magnitude can cross a BF16 rounding boundary for
            # many output elements at once. Mix two nearly identical input
            # amplitudes on that boundary; this keeps the requested flat
            # noise to sub-ULP precision before the unavoidable BF16 cast.
            lower_candidate = (ref + direction * low).to(torch.bfloat16)
            upper_candidate = (ref + direction * high).to(torch.bfloat16)
            lower_mse, _ = estimate(lower_candidate, reference)
            upper_mse, _ = estimate(upper_candidate, reference)
            require(lower_mse <= target_mse <= upper_mse and high > low,
                    "BF16 amplitude bracketing lost target")
            draws = torch.rand(reference.shape, device=reference.device,
                               generator=generator, dtype=torch.float32)
            fraction_low, fraction_high = 0.0, 1.0
            for _ in range(18):
                fraction = (fraction_low + fraction_high) / 2.0
                candidate = torch.where(draws < fraction, upper_candidate, lower_candidate)
                candidate_mse, _ = estimate(candidate, reference)
                gap = abs(candidate_mse / target_mse - 1)
                if gap < best_gap:
                    best_candidate, best_amplitude, best_gap = candidate, None, gap
                    row.update(amplitude_lower=low, amplitude_upper=high,
                               upper_amplitude_fraction=fraction)
                if gap <= 0.005:
                    break
                if candidate_mse < target_mse:
                    fraction_low = fraction
                else:
                    fraction_high = fraction
        modified = best_candidate
        row.update(random_seed=random_seed)
        if best_amplitude is not None:
            row["amplitude_after_bf16_correction"] = best_amplitude
    realized_mse, realized_nmse = estimate(modified, reference)
    row.update(realized_mse=realized_mse, realized_nmse=realized_nmse,
               relative_mse_change=realized_mse / target_mse - 1)
    require(bool(torch.isfinite(modified).all()), "intervention produced nonfinite output")
    if arm != "tail_restore":
        require(abs(row["relative_mse_change"]) <= 0.005,
                f"BF16-rounded {arm} could not match target MSE: {row['relative_mse_change']}")
    else:
        require(realized_mse <= target_mse + 1e-9,
                "tail restoration increased output error")
    return modified, row


def load_pair(*, full_main: bool):
    main = load_h3_pipeline(full=full_main, vram_limit_gib=18.0)
    main.load_models_to_device(["dit"])
    hooks = apply_svdquant(main.dit, STATE)
    reference = load_h3_pipeline(full=False, vram_limit_gib=18.0)
    reference.load_models_to_device(["dit"])
    main.dit.eval()
    reference.dit.eval()
    require(len(hooks) == 200, "SVDQuant target count mismatch")
    free, total = torch.cuda.mem_get_info()
    print(f"pair loaded: CUDA free {free/1024**3:.2f}/{total/1024**3:.2f} GiB", flush=True)
    require(free > 10 * 1024**3, "less than 10 GiB free after paired model loading")
    return main, reference, hooks


@torch.inference_mode()
def preflight(case: str) -> None:
    info = case_info(case)
    cached_path = Path("/data1/models/svdquant-wjq/research/20261015/E101_h3_r32_fixed1_three_cases") / case / "bf16_teacher_calls.pt"
    require(cached_path.is_file(), "missing BF16 teacher calls")
    calls = torch.load(cached_path, map_location="cpu", weights_only=False)
    require(len(calls) == 20, "incomplete E101 teacher calls")
    main, reference, hooks = load_pair(full_main=False)
    call = calls[0]
    args = tree_device(call["args"], "cuda")
    kwargs = tree_device(call["kwargs"], "cuda")
    ref_video, ref_audio = split_output(reference.dit(*args, **kwargs))
    q_video, q_audio = split_output(main.dit(*args, **kwargs))
    for name, ref, q, cached in (("video", ref_video, q_video, call["video"]),
                                 ("audio", ref_audio, q_audio, call["audio"])):
        ref_replay_mse, _ = estimate(ref, cached.to("cuda"))
        target_nmse = estimate(q, ref)[1]
        expected_nmse = info["e101"]["results"][0][name]["nmse"]
        require(ref_replay_mse < 1e-5 and abs(target_nmse - expected_nmse) < 1e-4,
                f"{name} first-step BF16/quant replay differs from E101")
        for i, arm in enumerate(ARMS):
            _, row = alter(ref, q, arm, info["seed"], 0, 0 if name == "video" else 1)
            print(f"preflight {case}/{name}/{arm}: target={target_nmse:.6f} "
                  f"realized={row['realized_nmse']:.6f}", flush=True)
    require(all(h.act.calls > 0 for h in hooks), "SVDQuant hooks were not exercised")
    print(f"PREFLIGHT PASS {case}: peak CUDA allocation {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)


@torch.inference_mode()
def denoise(case: str, arm: str) -> None:
    info = case_info(case)
    report_path = REPORT / case / f"{arm}_denoise.json"
    stage = DATA / case / arm
    terminal = stage / "final_latents.pt"
    require(not report_path.exists() and not terminal.exists(), "existing arm output; refusing overwrite")
    require(PLAN.is_file() and STATE.is_file(), "experiment plan or checkpoint missing")
    stage.mkdir(parents=True, exist_ok=True)
    report = {"format": DENOISE_FORMAT, "status": "running",
              "case": case, "arm": arm, "prompt_id": info["prompt_id"], "seed": info["seed"],
              "settings": {"height": 576, "width": 1024, "frames": 124, "steps": 20,
                           "cfg": 1.0, "video_flow_shift": 12.0, "audio_flow_shift": 3.0,
                           "rand_device": "cpu", "tiled": True, "fps": 24},
              "state_sha256": sha256(STATE), "plan_sha256": sha256(PLAN),
              "steps": [], "started_unix": time.time()}
    save_json(report_path, report)
    main, reference, hooks = load_pair(full_main=True)
    original_forward = main.dit.forward
    original_step = main.step
    original_loader = main.load_models_to_device
    final: dict[str, torch.Tensor] = {}

    def intervened_forward(_self, *args, **kwargs):
        step = len(report["steps"])
        require(step < 20, "more than 20 DiT calls")
        require(torch.cuda.mem_get_info()[0] > 5 * 1024**3,
                "less than 5 GiB free before paired DiT call")
        ref = split_output(reference.dit(*args, **kwargs))
        q = split_output(original_forward(*args, **kwargs))
        output = []
        entry = {"step": step, "modalities": {}}
        for modality_id, (name, r, quant) in enumerate(zip(("video", "audio"), ref, q, strict=True)):
            modified, row = alter(r, quant, arm, info["seed"], step, modality_id)
            output.append(modified)
            entry["modalities"][name] = row
            if step == 0:
                expected = info["e101"]["results"][0][name]["nmse"]
                require(abs(row["target_nmse"] - expected) < 1e-4,
                        f"first-step {name} Q/BF16 NMSE differs from E101 baseline")
        report["steps"].append(entry)
        save_json(report_path, report)
        print(f"{case}/{arm} step {step:02d}: video target={entry['modalities']['video']['target_nmse']:.6f} "
              f"realized={entry['modalities']['video']['realized_nmse']:.6f}", flush=True)
        return tuple(output)

    def observed_step(_self, scheduler, latents, progress_id, noise_pred, **kwargs):
        name = "video" if scheduler is main.scheduler else "audio" if scheduler is main.scheduler_audio else None
        require(name is not None and progress_id == len(report["steps"]) - 1,
                "scheduler step does not match current paired DiT output")
        updated = original_step(scheduler, latents, progress_id, noise_pred=noise_pred, **kwargs)
        require(bool(torch.isfinite(updated).all()), "nonfinite recurrent latents")
        if progress_id == 19:
            final[name + "_latents"] = updated.detach().cpu()
        return updated

    def stop_before_decode(_self, names):
        if list(names) == ["video_vae"] and len(report["steps"]) == 20:
            require(set(final) == {"video_latents", "audio_latents"}, "terminal latents incomplete")
            raise DenoiseComplete()
        return original_loader(names)

    main.dit.forward = types.MethodType(intervened_forward, main.dit)
    main.step = types.MethodType(observed_step, main)
    main.load_models_to_device = types.MethodType(stop_before_decode, main)
    try:
        try:
            main(prompt=info["prompt"], seed=info["seed"], height=576, width=1024,
                 num_frames=124, num_inference_steps=20, cfg_scale=1.0,
                 flow_shift=12.0, audio_flow_shift=3.0, rand_device="cpu", tiled=True)
        except DenoiseComplete:
            pass
        else:
            raise RuntimeError("expected to stop after 20 denoising steps, before VAE decode")
        require(len(report["steps"]) == 20 and all(h.act.calls > 0 for h in hooks),
                "incomplete paired calls or unused SVDQuant activation hook")
        with terminal.open("xb") as stream:
            torch.save(final, stream)
        report.update(status="denoise_complete", completed_unix=time.time(),
                      terminal_latents=str(terminal.resolve()), terminal_sha256=sha256(terminal),
                      terminal_bytes=terminal.stat().st_size,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        save_json(report_path, report)
        print(f"DENOISE COMPLETE {case}/{arm}: {terminal}", flush=True)
    except Exception as error:
        report.update(status="failed", failed_unix=time.time(), failure=repr(error))
        save_json(report_path, report)
        raise
    finally:
        del final, reference, main
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def decode(case: str, arm: str) -> None:
    from diffsynth.utils.data.audio_video import write_video_audio
    import av

    report_path = REPORT / case / f"{arm}_denoise.json"
    report = json.loads(report_path.read_text())
    require(report["status"] == "denoise_complete" and report["case"] == case and report["arm"] == arm,
            "validated complete denoising prerequisite missing")
    terminal = Path(report["terminal_latents"])
    require(sha256(terminal) == report["terminal_sha256"], "terminal latent checksum mismatch")
    video_path = DATA / case / arm / f"{case}_{arm}_576x1024_124f_20steps.mp4"
    media_report = REPORT / case / f"{arm}_video.json"
    require(not video_path.exists() and not media_report.exists(), "existing decoded output; refusing overwrite")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(path=local_h3_files("FL2VA/video_vae/source/model.safetensors"), **disk_config()),
            ModelConfig(path=local_h3_files("FL2VA/audio_vae/model.safetensors"), **disk_config()),
        ], processor_config=None, vram_limit=28.0)
    require(pipe.dit is None and pipe.video_vae is not None and pipe.audio_vae is not None,
            "decode process must contain only VAEs")
    latents = torch.load(terminal, map_location="cpu", weights_only=True)
    pipe.load_models_to_device(["video_vae"])
    decoded = pipe.video_vae.decode_video(latents["video_latents"].cuda(), dtype=torch.bfloat16,
                                          tiled=True, tile_size=256, tile_overlap=64)
    require(bool(torch.isfinite(decoded).all()), "nonfinite decoded video")
    video = pipe.vae_output_to_video(decoded, min_value=0, max_value=1)
    del decoded
    pipe.load_models_to_device(["audio_vae"])
    waveform = pipe.audio_vae.decode_audio(latents["audio_latents"].cuda(), dtype=torch.bfloat16)
    audio = pipe.output_audio_format_check(waveform)
    require(len(video) == 124 and all(frame.size == (1024, 576) for frame in video) and
            bool(torch.isfinite(audio).all()), "decoded media shape or finite check failed")
    write_video_audio(video=video, audio=audio, output_path=str(video_path), fps=24,
                      audio_sample_rate=pipe.audio_vae.sample_rate)
    with av.open(str(video_path)) as container:
        frames = sum(1 for _ in container.decode(video=0))
        video_stream = container.streams.video[0]
        require(frames == 124 and video_stream.width == 1024 and video_stream.height == 576 and
                len(container.streams.audio) == 1, "encoded media validation failed")
    media = {"format": VIDEO_FORMAT, "case": case, "arm": arm,
             "video": str(video_path.resolve()), "sha256": sha256(video_path),
             "bytes": video_path.stat().st_size, "frames": 124, "width": 1024, "height": 576,
             "fps": 24, "audio_streams": 1, "denoise_report": str(report_path.resolve()),
             "denoise_report_sha256": sha256(report_path)}
    save_json(media_report, media)
    print(f"VIDEO COMPLETE {case}/{arm}: {video_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "denoise", "decode"), required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    require((args.phase == "preflight") == (args.arm is None),
            "--arm is required only for denoise/decode")
    torch.set_num_threads(6)
    if args.phase == "preflight":
        preflight(args.case)
    elif args.phase == "denoise":
        denoise(args.case, args.arm)
    else:
        decode(args.case, args.arm)


if __name__ == "__main__":
    main()
