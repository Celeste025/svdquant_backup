#!/usr/bin/env python3
"""Create same-frame SVDQuant/FT contact sheets for E086 visual review."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def frames(path: Path, n: int = 5):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids = np.linspace(0, max(total - 1, 0), n, dtype=int)
    result = []
    for index in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {index} from {path}")
        frame = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, f"frame {index}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {index}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1, cv2.LINE_AA)
        result.append(frame)
    cap.release()
    return result


def labeled_row(label: str, images: list[np.ndarray]):
    strip = cv2.hconcat(images)
    label_area = np.full((42, strip.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(label_area, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2, cv2.LINE_AA)
    return cv2.vconcat([label_area, strip])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=int, nargs="+", default=[2, 16])
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    for prompt_id in args.prompts:
        source = args.input_dir / f"p{prompt_id}"
        svd = next(source.glob("svdquant_id*.mp4"))
        ft = next(source.glob("svdquant_ft_id*.mp4"))
        sheet = cv2.vconcat([labeled_row("SVDQuant", frames(svd)), labeled_row("SVDQuant + fine-tuning", frames(ft))])
        target = args.output_dir / f"p{prompt_id}_svdquant_vs_ft_contact.png"
        if not cv2.imwrite(str(target), sheet):
            raise RuntimeError(f"failed to write {target}")
        print(target)


if __name__ == "__main__":
    main()
