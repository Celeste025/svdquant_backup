import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, bridge, cpu
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/admin/workspace/aop_lab/app_data/models/VisionReward-Video"
QUESTIONS_PATH = Path("/tmp/VisionReward-repo/VisionReward_Video/VisionReward_video_qa.txt")
TORCH_TYPE = torch.bfloat16
METRIC_LABEL = "exploratory_unweighted_64q"

with QUESTIONS_PATH.open() as f:
    QUESTIONS = [line.strip() for line in f if line.strip()]

if len(QUESTIONS) != 64:
    raise RuntimeError(f"expected 64 full VisionReward questions, got {len(QUESTIONS)}")

QUESTION_MANIFEST_SHA256 = hashlib.sha256(
    "\n".join(QUESTIONS).encode("utf-8")
).hexdigest()

TOPICS = (
    ("prompt_alignment", range(0, 4)),
    ("composition_focus_camera", range(4, 10)),
    ("color", range(10, 14)),
    ("lighting", range(14, 20)),
    ("shape_consistency", range(20, 27)),
    ("object_motion", range(27, 31)),
    ("camera_motion", range(31, 35)),
    ("motion_smoothness_realism", range(35, 39)),
    ("clarity", range(39, 43)),
    ("stability", range(43, 49)),
    ("detail", range(49, 53)),
    ("text", range(53, 56)),
    ("physics", range(56, 59)),
    ("physical_world", range(59, 60)),
    ("safety", range(60, 64)),
)


def load_model(device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
    ).eval().to(device)
    return model, tokenizer


def load_video(video_data, strategy="chat"):
    bridge.set_bridge("torch")
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
    total_frames = len(decord_vr)
    timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
    timestamps = [timestamp[0] for timestamp in timestamps]
    frame_ids = []
    for second in range(round(max(timestamps)) + 1):
        closest_timestamp = min(timestamps, key=lambda timestamp: abs(timestamp - second))
        frame_ids.append(timestamps.index(closest_timestamp))
        if len(frame_ids) >= num_frames:
            break
    video = decord_vr.get_batch(frame_ids)
    return video.permute(3, 0, 1, 2)


def _prepare_inputs(model, tokenizer, video, query, device):
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=[],
        template_version="chat",
    )
    return {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
        "images": [[inputs["images"][0].to(device).to(TORCH_TYPE)]],
    }


def score_video_q64(model, tokenizer, video_path, prompt, device="cuda:0"):
    video = load_video(Path(video_path).read_bytes())
    yes_token_id = tokenizer.encode("Yes")[0]
    no_token_id = tokenizer.encode("No")[0]
    generated_text = []
    binary_yes = []
    probs_yes = []

    for question in QUESTIONS:
        query = question.replace("[[prompt]]", prompt)
        inputs = _prepare_inputs(model, tokenizer, video, query, device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
            p_yes = torch.softmax(logits[[yes_token_id, no_token_id]].float(), dim=0)[0].item()
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                pad_token_id=128002,
                top_k=1,
                do_sample=False,
                top_p=0.1,
                temperature=0.1,
            )
        first_generated_token = outputs[0, inputs["input_ids"].shape[1]]
        answer = tokenizer.decode(first_generated_token).strip().lower()
        generated_text.append(answer)
        binary_yes.append(answer == "yes")
        probs_yes.append(p_yes)

    return {
        "generated_text": generated_text,
        "binary_yes": binary_yes,
        "p_yes_vs_no": probs_yes,
        "unweighted_64q_binary_yes_rate": float(np.mean(binary_yes)),
        "unweighted_64q_mean_p_yes": float(np.mean(probs_yes)),
    }
