#!/usr/bin/env python3
"""Score generated videos with the official VisionReward-Video checklist.

The input files are the per-case generation status JSON files already stored
alongside samples.  Scores preserve the official weighted +/-1 formulation
while retaining every answer for inspection.
"""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, bridge, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_video(path: str):
    bridge.set_bridge("torch")
    reader = VideoReader(io.BytesIO(Path(path).read_bytes()), ctx=cpu(0))
    timestamps = [item[0] for item in reader.get_frame_timestamp(np.arange(len(reader)))]
    indices = []
    for second in range(round(max(timestamps)) + 1):
        indices.append(min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - second)))
        if len(indices) == 24:
            break
    return reader.get_batch(indices).permute(3, 0, 1, 2), indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--case-json", action="append", required=True,
                        help="generation status JSON; repeat for each video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int, default=0,
                        help="nonzero only for a smoke test")
    args = parser.parse_args()

    questions = [line.strip() for line in Path(args.questions).read_text().splitlines() if line.strip()]
    weights = np.asarray(json.loads(Path(args.weights).read_text()), dtype=float)
    if len(questions) != len(weights):
        raise ValueError(f"question/weight mismatch: {len(questions)} vs {len(weights)}")
    if args.max_questions:
        questions, weights = questions[:args.max_questions], weights[:args.max_questions]

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    ).eval().to("cuda")
    yes_token = tokenizer.encode("Yes")[0]
    records = []
    for case_path in args.case_json:
        case = json.loads(Path(case_path).read_text())
        video, frame_indices = load_video(case["video"])
        answers = []
        for question in questions:
            query = question.replace("[[prompt]]", case["prompt"])
            payload = model.build_conversation_input_ids(
                tokenizer=tokenizer, query=query, images=[video], history=[], template_version="chat"
            )
            inputs = {
                "input_ids": payload["input_ids"].unsqueeze(0).cuda(),
                "token_type_ids": payload["token_type_ids"].unsqueeze(0).cuda(),
                "attention_mask": payload["attention_mask"].unsqueeze(0).cuda(),
                "images": [[payload["images"][0].cuda().to(dtype)]],
            }
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=1, pad_token_id=128002, top_k=1,
                    do_sample=False, top_p=0.1, temperature=0.1,
                )[:, inputs["input_ids"].shape[1]]
            token_id = int(output[0])
            answer = tokenizer.decode(token_id).strip()
            answers.append({"question": question, "answer": answer, "yes": token_id == yes_token})
        signed = np.asarray([1.0 if answer["yes"] else -1.0 for answer in answers])
        sample_set = Path(case_path).parents[2].name
        records.append({
            "label": f"{sample_set}:{case.get('variant', Path(case_path).stem)}",
            "case_json": str(Path(case_path).resolve()),
            "video": case["video"], "prompt": case["prompt"], "frame_indices": frame_indices,
            "score": float(np.mean(signed * weights)), "answers": answers,
        })
    output = {"metric": "VisionReward-Video official selected-question weighted score",
              "model": str(Path(args.model_path).resolve()), "questions": len(questions), "results": records}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for row in records:
        print(f"{row['label']}: {row['score']:.8f}")


if __name__ == "__main__":
    main()
