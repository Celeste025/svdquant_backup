"""VisionReward-Video probability-weighted scoring: use P(yes) instead of binary yes/no."""
import io
import json
import numpy as np
import torch
from decord import cpu, VideoReader, bridge

import visionreward_common as vr


def prob_score_video(model, tokenizer, video_path, prompt, device='cuda'):
    yes_id = tokenizer.encode("Yes")[0]
    no_id = tokenizer.encode("No")[0]
    queries = [q.replace('[[prompt]]', prompt) for q in vr.QUESTIONS]
    probs = []
    for query in queries:
        video_data = open(video_path, 'rb').read()
        video = vr.load_video(video_data)
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer, query=query, images=[video], history=[],
            template_version='chat')
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
            'images': [[inputs['images'][0].to(device).to(vr.TORCH_TYPE)]],
        }
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1]
        p_yes = torch.softmax(logits[[yes_id, no_id]].float(), dim=0)[0].item()
        probs.append(p_yes)
    probs = np.array(probs)
    # 官方公式: mean(answer * weight), answer∈{+1,-1}; 概率版: answer = 2*P(yes)-1 ∈ [-1,1]
    answers = 2 * probs - 1
    return np.mean(answers * vr.WEIGHT).item(), probs.tolist()
