import io
import json
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/admin/workspace/aop_lab/app_data/models/VisionReward-Video"
QUESTIONS_PATH = "/tmp/VisionReward-repo/VisionReward_Video/VisionReward_video_qa_select.txt"
WEIGHT_PATH = "/tmp/VisionReward-repo/VisionReward_Video/weight.json"

with open(QUESTIONS_PATH, 'r') as f:
    QUESTIONS = [l.strip() for l in f if l.strip()]
with open(WEIGHT_PATH, 'r') as f:
    WEIGHT = np.array(json.load(f))

TORCH_TYPE = torch.bfloat16

def load_model(device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    ).eval().to(device)
    return model, tokenizer

def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(video_data), ctx=cpu(0))
    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data

def inference(model, tokenizer, video_path, query, device='cuda', temperature=0.1):
    video_data = open(video_path, 'rb').read()
    strategy = 'chat'
    video = load_video(video_data, strategy=strategy)
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=[],
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 8,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]]
    return tokenizer.decode(outputs[0]).strip().lower()

def score_video(model, tokenizer, video_path, prompt, device='cuda'):
    queries = [q.replace('[[prompt]]', prompt) for q in QUESTIONS]
    answers = []
    for query in queries:
        answer = inference(model, tokenizer, video_path, query, device=device)
        answers.append(answer)
    answers = np.array([1 if answer == 'yes' else -1 for answer in answers])
    return np.mean(answers * WEIGHT).item(), answers.tolist()
