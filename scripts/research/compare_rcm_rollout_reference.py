#!/usr/bin/env python3
"""E022 fixed original-rCM architecture rollout, controlled BF16 FLASH SDPA.

Two existing trajectories, eight DiT forwards, four FP32 VAE decodes. No RNG
draws, quantization, new text encoding, downloads, or edits to previous runs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/data1/models/svdquant-wjq')
RCM_REPO=Path('/home/wjq/workspace/rcm')
BASE=DATA/'models/Wan2.1-T2V-1.3B-Diffusers'
CHECKPOINT=DATA/'models/rcm-Wan/rCM_Wan2.1_T2V_1.3B_480p.pt'
RD=ROOT/'results/research/E022'
GENERATION=RD/'video_baselines.json'
CASES=('vbench_197_r0','vbench_133_r0')
FRAMES=(0,19,38,57,76)
PRECISION_POLICY='Original source FP32-required time_embedding, time_projection, Head and affine WanLayerNorm parameters are explicitly FP32; other backbone weights/inputs and QKV attention are BF16. Original-source velocity remains its actual output dtype. This is a documented runtime compatibility policy, not a verbatim execution of the public all-BF16 loader.'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def record(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def tensor_sha(x,torch):
    return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def inputs(torch):
    generation=json.loads(GENERATION.read_text())
    require(generation['status']=='complete','Completed E022 baselines required')
    rows={};payloads={}
    for key in CASES:
        row=generation['arms']['bf16']['trajectories'][key]
        shared=generation['shared_inputs'][key]
        for item in (row['final_latent'],shared['artifact']):
            require(sha(item['path'])==item['sha256'],f'Saved input changed: {item["path"]}')
        require(row['shared_inputs']==shared['artifact']['path'],'Mismatched shared inputs')
        payload=torch.load(row['shared_inputs'],map_location='cpu',weights_only=False)
        require(payload['trajectory_id']==key and payload['prompt']==row['prompt'] and payload['seed']==row['seed'],'Input identity mismatch')
        x=payload['initial_latent_fp64'];embedding=payload['embedding_bf16'];noises=payload['update_noises_fp32']
        require(x.shape==(1,16,20,60,104) and x.dtype==torch.float64,'Wrong initial state')
        require(embedding.shape==(1,512,4096) and embedding.dtype==torch.bfloat16,'Wrong embedding')
        require(len(noises)==4 and all(n.shape==x.shape and n.dtype==torch.float32 for n in noises),'Wrong saved noises')
        require(tensor_sha(x,torch)==shared['initial_latent_sha256'] and tensor_sha(embedding,torch)==shared['embedding_sha256'],'Shared state/embedding mismatch')
        require([tensor_sha(n,torch) for n in noises]==shared['noise_sha256'],'Saved noises mismatch')
        steps=payload['t_steps_fp64']
        require(steps.dtype==torch.float64 and steps.tolist()==generation['schedule']['t_steps'],'Schedule mismatch')
        require(all(bool(torch.isfinite(t).all()) for t in (x,embedding,*noises,steps)),'Nonfinite saved input')
        rows[key]=row;payloads[key]=payload
    return generation,rows,payloads


def import_native():
    sys.path.insert(0,str(RCM_REPO))
    from rcm.networks.wan2pt1 import WanModel,WanLayerNorm
    return WanModel,WanLayerNorm


def prepare_precision(model,WanLayerNorm):
    model.to(dtype=__import__('torch').bfloat16)
    for module in (model.time_embedding,model.time_projection,model.head):module.float()
    for module in model.modules():
        if isinstance(module,WanLayerNorm):module.float()


def install_attention(model,torch,counts,check_budget,cpu_check=False):
    from torch.nn.attention import SDPBackend,sdpa_kernel
    class ControlledAttention(torch.nn.Module):
        def forward(self,q,k,v,*args,attn_ctx=None,attn_meta=None,**kwargs):
            require(not args and not kwargs and attn_meta is None,'Unexpected dense-attention arguments')
            require(q.ndim==k.ndim==v.ndim==4 and q.dtype==k.dtype==v.dtype==torch.bfloat16,'Expected BF16 [B,S,H,D] QKV')
            if attn_ctx is not None:
                require(attn_ctx.kv_cache is None and attn_ctx.attn_observer is None,'Unexpected KV cache or observer')
            check_budget()
            with sdpa_kernel(SDPBackend.MATH if cpu_check else SDPBackend.FLASH_ATTENTION):
                out=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                    attn_mask=None,dropout_p=0.,is_causal=False,scale=None)
            counts['sdpa_calls']+=1
            return out.transpose(1,2).contiguous()
    for block in model.blocks:
        block.self_attn.attn_op.local_attn=ControlledAttention()
        block.cross_attn.attn_op.local_attn=ControlledAttention()
    # The original module provides this PyTorch fallback when flash-attn is absent.
    import rcm.utils.rope as rope
    rope.flash_apply_rotary_emb=None


def check_only(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    generation,rows,payloads=inputs(torch)
    WanModel,WanLayerNorm=import_native()
    # Read only .pt metadata via mmap; no full-size CPU/GPU model computation.
    raw=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    source=raw.get('state_dict',raw)
    state={k.removeprefix('net.'):v for k,v in source.items() if not k.removeprefix('net.').startswith('accum_')}
    with torch.device('meta'):
        model=WanModel(dim=1536,eps=1e-6,ffn_dim=8960,freq_dim=256,in_dim=16,model_type='t2v',num_heads=12,num_layers=30,out_dim=16,text_len=512)
    expected=model.state_dict()
    require(set(state)==set(expected),'Original checkpoint key mismatch')
    require(all(state[k].shape==expected[k].shape for k in state),'Original checkpoint shape mismatch')
    del model,expected,raw,source,state
    # A tiny untrained CPU model checks only dtype/layout/runtime compatibility.
    torch.set_num_threads(2)
    toy=WanModel(dim=48,eps=1e-6,ffn_dim=96,freq_dim=16,in_dim=16,model_type='t2v',num_heads=1,num_layers=1,out_dim=16,text_len=512,text_dim=32).eval()
    prepare_precision(toy,WanLayerNorm)
    counts=dict(sdpa_calls=0);install_attention(toy,torch,counts,lambda:None,cpu_check=True)
    with torch.inference_mode():
        y=toy(x_B_C_T_H_W=torch.zeros(1,16,1,2,2,dtype=torch.bfloat16),timesteps_B_T=torch.tensor([[988.]],dtype=torch.bfloat16),crossattn_emb=torch.zeros(1,512,32,dtype=torch.bfloat16))
    require(y.shape==(1,16,1,2,2) and bool(torch.isfinite(y).all()) and counts['sdpa_calls']==2,'CPU runtime preflight failed')
    require(not torch.cuda.is_initialized(),'CPU check initialized CUDA')
    compile(Path(__file__).read_text(),str(Path(__file__)),'exec')
    result=dict(status='complete',phase='CPU_prepare',cuda_initialized=False,source=record(__file__),
        cases={key:dict(prompt=row['prompt'],seed=row['seed'],shared_inputs=row['shared_inputs'],converted_final=row['final_latent']) for key,row in rows.items()},
        planned_dit_calls=8,planned_sdpa_calls=480,planned_vae_decodes=4,gpu=0,wall_seconds=args.wall_seconds,
        checkpoint_metadata=dict(path=str(CHECKPOINT),bytes=CHECKPOINT.stat().st_size),original_checkpoint_keys_and_shapes_match=True,
        original_umt5='Unavailable: only incomplete original .pth found. No load/download/reconstruction; original and converted paths share E022 cached real embeddings.',
        precision_policy=PRECISION_POLICY,cpu_toy_output_dtype=str(y.dtype),cpu_toy_attention='MATH only; GPU runner forces FLASH',
        scope='Controlled original architecture and original weights with fixed existing conditioning/noises, not complete original-UMT5 end-to-end reproduction.')
    require(not args.check_output.exists(),'Preserve existing CPU check')
    save_json(args.check_output,result);print(json.dumps(dict(status='complete',output=str(args.check_output),cuda_initialized=False)),flush=True)


def supervise(args):
    supervisor=RD/'rcm_rollout_reference_supervisor.json';log=ROOT/'results/logs/E022_rcm_rollout_reference.log'
    require(not any(p.exists() for p in (supervisor,log,args.output,args.artifact_dir)),'Preserve prior outputs')
    require(0<args.wall_seconds<=900,'Maximum supervisor budget is 15 minutes')
    start=time.time();deadline=start+args.wall_seconds
    command=[sys.executable,str(Path(__file__).resolve()),'--deadline-unix',str(deadline),'--output',str(args.output),'--artifact-dir',str(args.artifact_dir)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    log.parent.mkdir(parents=True,exist_ok=True)
    status=dict(status='starting',gpu=0,deadline_epoch=deadline,start_epoch=start,command=command,log=str(log),source=record(__file__))
    save_json(supervisor,status)
    with log.open('x') as stream:
        worker=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        status.update(status='running',pid=worker.pid);save_json(supervisor,status)
        timed_out=False
        try:code=worker.wait(timeout=max(.01,deadline-time.time()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(worker.pid,signal.SIGTERM)
            try:code=worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGKILL);code=worker.wait()
    status.update(status='deadline_terminated' if timed_out else ('complete' if code==0 else 'failed'),returncode=code,seconds=time.time()-start)
    save_json(supervisor,status);print(json.dumps(status),flush=True)
    if code:raise SystemExit(code)


def execute(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','This fixed diagnostic is assigned GPU0')
    require(args.deadline_unix and args.deadline_unix>time.time(),'Explicit future deadline required')
    require(not args.output.exists() and not args.artifact_dir.exists(),'Preserve all old artifacts')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    import numpy as np
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from diffusers.utils import export_to_video
    from PIL import Image,ImageDraw,ImageFont
    sys.path.insert(0,str(ROOT/'scripts'))
    from infer_rcm_wan_4step import decode_spatial_tiled
    require(torch.cuda.get_device_capability()==(12,0),'Expected SM120')
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(60*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    generation,rows,payloads=inputs(torch);WanModel,WanLayerNorm=import_native()
    args.artifact_dir.mkdir(parents=True);started=time.time()
    counts=dict(dit_calls=0,sdpa_calls=0,vae_decodes=0)
    report=dict(status='running',cases={},actual_counts=counts,source=record(__file__),generation=record(GENERATION),
        rcm_head=subprocess.check_output(['git','-C',str(RCM_REPO),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_metadata=dict(path=str(CHECKPOINT),bytes=CHECKPOINT.stat().st_size,mtime_ns=CHECKPOINT.stat().st_mtime_ns),
        attention='Original architecture QKV/RoPE paths; controlled BF16 PyTorch FLASH SDPA, no original flash-attn binary claim.',
        attention_contract=dict(layout='B,S,H,D -> B,H,S,D',mask=None,dropout_p=0.,is_causal=False,scale='default 1/sqrt(head_dim)',rope='original PyTorch FP32 fallback'),
        precision_policy=PRECISION_POLICY,conditioning='E022 cached real embedding used unchanged by both implementations. Original UMT5 checkpoint incomplete; original text encoder not tested.',
        normalization='Both final latents: FP64 -> FP32, z/(1/config.latents_std)+config.latents_mean; same FP32 VAE, core128 halo0 single tile.',
        frame_indices=list(FRAMES),limits=dict(gpu=0,wall_seconds=900,memory_gib=60),
        interpretation='Architecture/runtime-path diagnostic; no arbitrary byte/NMSE quality gate and no single-cause conversion claim.')

    def budget():
        if time.time()>=args.deadline_unix:raise TimeoutError('Reference rollout deadline expired')
        if torch.cuda.max_memory_allocated()>60*1024**3:raise MemoryError('60GiB allocation guard exceeded')

    def checkpoint():
        report.update(seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        save_json(args.output,report);budget()

    def save_tensor(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path);return record(path)

    def difference(x,y):
        x=x.double();y=y.double();delta=x-y
        return dict(mae=float(delta.abs().mean()),rmse=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),
            nmse_vs_converted=float(delta.square().sum()/y.square().sum().clamp_min(1e-30)))

    frame_images={}
    try:
        budget()
        raw=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
        state={k.removeprefix('net.'):v for k,v in raw.get('state_dict',raw).items() if not k.removeprefix('net.').startswith('accum_')}
        with torch.device('meta'):
            model=WanModel(dim=1536,eps=1e-6,ffn_dim=8960,freq_dim=256,in_dim=16,model_type='t2v',num_heads=12,num_layers=30,out_dim=16,text_len=512)
        model.load_state_dict(state,strict=True,assign=True);del state,raw
        prepare_precision(model,WanLayerNorm);model=model.cuda().eval().requires_grad_(False)
        report['parameter_dtypes']={name:str(value.dtype) for name,value in model.named_parameters()}
        install_attention(model,torch,counts,budget)
        with torch.inference_mode():
            for key in CASES:
                payload=payloads.pop(key);row=rows[key];folder=args.artifact_dir/key
                x=payload['initial_latent_fp64'].cuda();embedding=payload['embedding_bf16'].cuda();steps=payload['t_steps_fp64'].cuda()
                entry=dict(prompt=row['prompt'],seed=row['seed'],shared_inputs=record(row['shared_inputs']),initial_latent_sha256=tensor_sha(x,torch),
                    embedding_sha256=tensor_sha(embedding,torch),steps=[],converted_final=row['final_latent'])
                report['cases'][key]=entry
                entry['embedding']=save_tensor(folder/'shared_embedding.pt',payload['embedding_bf16'])
                entry['embedding_comparison']=dict(source='Exact saved E022 tensor',equal=True,mae=0.,rmse=0.)
                ones=torch.ones(1,1,device='cuda',dtype=torch.float64)
                for index,(t_cur,t_next) in enumerate(zip(steps[:-1],steps[1:])):
                    budget();before=x.cpu();timestep=(t_cur.float()*ones*1000).to(torch.bfloat16);count_before=counts['sdpa_calls'];begin=time.time()
                    velocity=model(x_B_C_T_H_W=x.to(torch.bfloat16),timesteps_B_T=timestep,crossattn_emb=embedding)
                    counts['dit_calls']+=1
                    noise=payload['update_noises_fp32'][index].cuda()
                    x=(1-t_next)*(x-t_cur*velocity.to(torch.float64))+t_next*noise
                    require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(velocity).all()),'Nonfinite native rollout')
                    torch.cuda.synchronize()
                    step=dict(step=index,timestep=float(timestep.item()),velocity_dtype=str(velocity.dtype),actual_sdpa_calls=counts['sdpa_calls']-count_before,
                        seconds=time.time()-begin,noise_sha256=tensor_sha(noise,torch),state_before_sha256=tensor_sha(before,torch),
                        velocity_sha256=tensor_sha(velocity,torch),state_after_sha256=tensor_sha(x,torch))
                    step['artifact']=save_tensor(folder/f'step_{index:02d}.pt',dict(state_before_fp64=before,actual_model_input_bf16=before.to(torch.bfloat16),
                        timestep_bf16=timestep.cpu(),velocity=velocity.cpu(),state_after_fp64=x.cpu(),noise_sha256=step['noise_sha256']))
                    require(step['actual_sdpa_calls']==60,'Expected 60 attention calls per DiT')
                    entry['steps'].append(step);checkpoint()
                    del before,velocity,noise
                entry['native_final']=save_tensor(folder/'native_final_latent.pt',x.cpu());entry['native_final_tensor_sha256']=tensor_sha(x,torch)
                converted=torch.load(row['final_latent']['path'],map_location='cpu',weights_only=True)
                require(tensor_sha(converted,torch)==row['final_latent_sha256'],'Converted final tensor changed')
                entry['final_difference_vs_converted']=difference(x.cpu(),converted)
                del converted,payload,x,embedding,steps;checkpoint()
            del model;gc.collect();torch.cuda.empty_cache()
            vae=AutoencoderKLWan.from_pretrained(BASE/'vae',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
            mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,16,1,1,1)
            reciprocal_std=1.0/torch.tensor(vae.config.latents_std,device='cuda').view(1,16,1,1,1)
            processor=VideoProcessor(vae_scale_factor=8)
            for key in CASES:
                entry=report['cases'][key];entry['decodes']={};native_pixels=None
                for arm,item in [('native',entry['native_final']),('converted',entry['converted_final'])]:
                    budget();folder=args.artifact_dir/key/arm;folder.mkdir(parents=True)
                    latent=torch.load(item['path'],map_location='cpu',weights_only=True)
                    normalized=latent.cuda().float()/reciprocal_std+mean;vae.clear_cache();begin=time.time()
                    try:decoded=decode_spatial_tiled(vae,normalized,core=128,halo=0)
                    finally:vae.clear_cache()
                    torch.cuda.synchronize();counts['vae_decodes']+=1
                    require(decoded.shape==(1,3,77,480,832) and bool(torch.isfinite(decoded).all()),'Invalid full video decode')
                    rgb=(decoded[0]+1)*.5
                    result=dict(seconds=time.time()-begin,decoded_tensor_sha256=tensor_sha(decoded,torch),pixel_stats=dict(
                        rgb_mean=rgb.mean(dim=(1,2,3)).tolist(),rgb_std=rgb.std(dim=(1,2,3),unbiased=False).tolist(),
                        column_rgb_mean=rgb.mean(dim=(1,2)).tolist(),per_frame_rgb_mean=rgb.mean(dim=(2,3)).t().tolist()))
                    if arm=='native':native_pixels=decoded
                    else:entry['decoded_difference_native_vs_converted']=difference(native_pixels,decoded)
                    frames=processor.postprocess_video(decoded,output_type='np')[0]
                    video=folder/'video.mp4';export_to_video(frames,str(video),fps=16)
                    result['video']=record(video)|dict(frames=77,height=480,width=832,fps=16);result['saved_frames']=[]
                    for index in FRAMES:
                        image=Image.fromarray((frames[index]*255).round().clip(0,255).astype(np.uint8))
                        path=folder/f'frame_{index:03d}.png';image.save(path);result['saved_frames'].append(record(path)|dict(frame_index=index))
                        frame_images[(key,arm,index)]=image.resize((416,240),Image.Resampling.LANCZOS)
                    entry['decodes'][arm]=result
                    del latent,normalized,decoded,rgb,frames;gc.collect();torch.cuda.empty_cache();checkpoint()
                del native_pixels
            del vae;gc.collect();torch.cuda.empty_cache()
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
        except OSError:font=ImageFont.load_default(size=21)
        report['contact_sheets']=[]
        for key in CASES:
            sheet=Image.new('RGB',(2360,670),(247,248,250));draw=ImageDraw.Draw(sheet)
            draw.text((15,15),f'{key} | same cached text + actual noise | {rows[key]["prompt"]}',font=font,fill='black')
            for col,index in enumerate(FRAMES):draw.text((230+422*col,63),f'frame {index}',font=font,fill='black')
            for row,arm in enumerate(('native','converted')):
                y=110+260*row;draw.text((15,y+80),f'{arm}\nFP32 VAE',font=font,fill='black')
                for col,index in enumerate(FRAMES):sheet.paste(frame_images[(key,arm,index)],(230+422*col,y))
            draw.text((15,637),'Original architecture uses explicit FP32-required stages + BF16 FLASH SDPA; not an original UMT5 end-to-end test.',font=font,fill='black')
            path=args.artifact_dir/f'{key}_reference.png';sheet.save(path);report['contact_sheets'].append(record(path))
        require(counts==dict(dit_calls=8,sdpa_calls=480,vae_decodes=4),'Unexpected execution counts')
        report['status']='complete';checkpoint()
    except BaseException:
        report.update(status='failed_partial_preserved',error=traceback.format_exc());save_json(args.output,report);raise
    print(json.dumps(dict(status='complete',output=str(args.output),actual_counts=counts)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true');mode.add_argument('--supervise',action='store_true')
    parser.add_argument('--wall-seconds',type=int,default=900);parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--check-output',type=Path,default=RD/'rcm_rollout_reference_cpucheck.json')
    parser.add_argument('--output',type=Path,default=RD/'rcm_rollout_reference.json')
    parser.add_argument('--artifact-dir',type=Path,default=DATA/'research/20261003/E022/rcm_rollout_reference')
    args=parser.parse_args()
    if args.check_only:check_only(args)
    elif args.supervise:supervise(args)
    else:execute(args)


if __name__=='__main__':main()
