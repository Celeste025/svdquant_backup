#!/usr/bin/env python3
"""Capture real first/later-step QKV for same-layer route-reuse validation."""
import hashlib, importlib, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from minimax_h3_svdquant_common import load_h3_pipeline, read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy
DATA=Path('/data1/models/svdquant-wjq/research/20261102/E187_h3_480p_trajectory')
REPORT=ROOT/'results/research/E187'
class CaptureComplete(Exception):pass
@torch.inference_mode()
def main():
    assert not (REPORT/'manifest.json').exists()
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
    row=read_prompt(16);pipe=load_h3_pipeline(full=True,vram_limit_gib=18.)
    pipe.load_models_to_device(['dit']);pipe.dit.eval()
    grid=(37,15,26);state={'step':-1,'layer':None};captures=[];hooks=[]
    def bind(module,args,kwargs):
        state['step']+=1;state['layer']=None
        ids=kwargs['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long()
        state['prefix']=int(ids[0]);assert ids.numel()==37*15*26
        assert torch.equal(ids,torch.arange(state['prefix'],state['prefix']+ids.numel()))
    hooks.append(pipe.dit.register_forward_pre_hook(bind,with_kwargs=True))
    def enter(layer):
        def callback(*args):state['layer']=layer
        return callback
    def exit_layer(*args):state['layer']=None
    for layer in (0,25,49):
        hooks.append(pipe.dit.blocks[layer].attn.register_forward_pre_hook(enter(layer)))
        hooks.append(pipe.dit.blocks[layer].attn.register_forward_hook(exit_layer))
    original=comfy._sdpa_varlen_attention
    def intercept(q,k,v,cu_seqlens,softmax_scale):
        layer=state['layer'];step=state['step']
        if layer in (0,25,49) and step in (0,10):
            bounds=cu_seqlens.detach().cpu().tolist();n=bounds[1]
            assert n==state['prefix']+37*15*26 and q.shape==k.shape==v.shape and q.dtype==torch.bfloat16
            path=DATA/f'race_480p_124f_step{step}_block{layer}.pt'
            packet={name:x[:n].transpose(0,1).unsqueeze(0).cpu().contiguous() for name,x in zip(('q','k','v'),(q,k,v))}
            packet.update(prefix_tokens=state['prefix'],video_grid=grid,layer=layer,step=step,
                          case='race',frames=124,height=480,width=832,bounds=bounds,softmax_scale=float(softmax_scale))
            with path.open('xb') as f:torch.save(packet,f)
            captures.append({'step':step,'layer':layer,'path':str(path),'shape_hnd':list(packet['q'].shape)})
            print(f'CAPTURED {path} shape={tuple(packet["q"].shape)}',flush=True)
            del packet
            if step==10 and layer==49:raise CaptureComplete()
        return original(q,k,v,cu_seqlens,softmax_scale)
    comfy._sdpa_varlen_attention=intercept
    try:
        try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,
                 num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
        except CaptureComplete:pass
        else:raise RuntimeError('target capture stop not reached')
    finally:
        comfy._sdpa_varlen_attention=original
        for hook in hooks:hook.remove()
    assert {(x['step'],x['layer']) for x in captures}=={(s,l) for s in (0,10) for l in (0,25,49)}
    for entry in captures:
        digest=hashlib.sha256()
        with Path(entry['path']).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        entry['sha256']=digest.hexdigest()
    report={'status':'complete','purpose':'step0 route capture / step10 actual reuse input',
            'prompt_id':16,'seed':int(row['seed']),'settings':{'height':480,'width':832,'frames':124,
            'steps':20,'cfg':1.,'flow_shift':12.,'audio_flow_shift':3.,'rand_device':'cpu'},'captures':captures}
    (REPORT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('E187_CAPTURE_COMPLETE',flush=True)
if __name__=='__main__':main()
