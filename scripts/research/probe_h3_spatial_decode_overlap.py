#!/usr/bin/env python3
"""E076 bounded first-window original VAE raw-tile probe."""
import os,time,json,hashlib,traceback
from pathlib import Path
import torch
import av
from PIL import Image,ImageDraw
import run_h3_phase_suffix_v2 as prior
old=prior.old
R=Path(__file__).resolve().parents[2];D=R/'results/research/E076';DATA=Path('/data1/models/svdquant-wjq/research/20261004/E076')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgb(vae,x):return vae.processor.revert_tensor(x.float()).clamp(0,1)
def pic(x):return Image.fromarray(x.permute(1,2,0).mul(255).round().byte().numpy())
@torch.inference_mode()
def main():
 D.mkdir(exist_ok=False);DATA.mkdir(exist_ok=False);t=time.monotonic();deadline=t+170
 report={'status':'running','decoder_calls':0,'driver_sha256':sha(__file__),'cases':[]}
 torch.set_num_threads(4)
 try:
  pipe=old.MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda',model_configs=[old.ModelConfig(path=str(old.MODEL_ROOT/'FL2VA/video_vae/source/model.safetensors'),**old.disk_config())],processor_config=None,vram_limit=24.)
  pipe.load_models_to_device(['video_vae']);vae=pipe.video_vae;vae.eval()
  original=vae.decode;raw=[]
  def capture(z):
   assert time.monotonic()<deadline and report['decoder_calls']<30
   assert torch.cuda.max_memory_allocated()<30*1024**3
   y=original(z);raw.append(y[:,:,:20].cpu());report['decoder_calls']+=1
   return y
  vae.decode=capture
  for arm,source in [('bf16','results/research/E073/denoise_full_bf16_r1.json'),('svd','results/research/E038/denoise_bf16.json')]:
   c=next(c for c in json.loads((R/source).read_text())['cases'] if c['prompt_id']==161 and c['replica']==1);rec=c['final_latents'];assert sha(rec['file'])==rec['sha256'];v=torch.load(rec['file'],map_location='cpu',weights_only=True)['video_latents']
   from diffsynth.models.minimax_h3_video_vae import _VIDEO_LATENTS_MEAN,_VIDEO_LATENTS_STD
   mean=torch.tensor(_VIDEO_LATENTS_MEAN).reshape(1,-1,1,1,1);std=torch.tensor(_VIDEO_LATENTS_STD).reshape(1,-1,1,1,1)
   z=(v[:,:,:7].float()*std+mean).bfloat16().cuda();vae.tile_size=256;vae.tile_overlap_min=64
   ys,yl,yo=vae.split_tiles(576);xs,xl,xo=vae.split_tiles(1024);raw.clear();out=vae.tiled_decode(z);torch.cuda.synchronize();assert len(raw)==15
   merged=rgb(vae,out[:,:,:20].cpu())[0,:,3:20];tiles=[rgb(vae,x)[0,:,3:20] for x in raw]
   p=DATA/(arm+'_first_window.pt');torch.save({'tiles':raw,'merged':out[:,:,:20].cpu(),'ys':ys,'xs':xs},p)
   stats=[]
   for i in range(len(ys)):
    for j in range(len(xs)):
     k=i*len(xs)+j
     for dim,condition,other,overlap in [('x',j+1<len(xs),k+1,xo[j] if j<len(xo) else 0),('y',i+1<len(ys),k+len(xs),yo[i] if i<len(yo) else 0)]:
      if not condition:continue
      a=tiles[k];b=tiles[other]
      if dim=='x':a=a[:,:,:,-overlap:];b=b[:,:,:,:overlap]
      else:a=a[:,:,-overlap:,:];b=b[:,:,:overlap,:]
      delta=(a-b).double();stats.append({'tile':k,'neighbor':other,'dim':dim,'overlap':overlap,'mae':float(delta.abs().mean()),'rmse':float(delta.square().mean().sqrt())})
   # Every raw tile shown at each fixed frame; no selected mask or discarded tile.
   for f in [0,4,8,12]:
    canvas=Image.new('RGB',(1280,3*278+310),'white');draw=ImageDraw.Draw(canvas)
    for k,tile in enumerate(tiles):
     i,j=divmod(k,5);canvas.paste(pic(tile[:,f]),(j*256,i*278));draw.text((j*256+3,i*278+258),f'{arm} tile{k} x{xs[j]} y{ys[i]} f{f}',fill='black')
    canvas.paste(pic(merged[:,f]).resize((512,288)),(0,834));draw.text((520,850),'Original spatial blend; all raw tiles above',fill='black');canvas.save(D/f'{arm}_frame{f:03d}.png')
   decode_source='results/research/E073/decode_full_bf16_r1.json' if arm=='bf16' else 'results/research/E038/decode_bf16.json'
   dc=next(c for c in json.loads((R/decode_source).read_text())['cases'] if c['prompt_id']==161 and c['replica']==1)
   vr=dc['video'];assert sha(vr['file'])==vr['sha256'];diff=[]
   with av.open(vr['file']) as container:
    for f,frame in enumerate(container.decode(video=0)):
     if f==17:break
     a=torch.from_numpy(frame.to_ndarray(format='rgb24').copy()).permute(2,0,1).float()/255
     diff.append(float((a-merged[:,f]).abs().mean()))
   report['cases'].append({'arm':arm,'input':rec,'source_video':vr,'tiles':len(raw),'xs':xs,'ys':ys,'raw_artifact':{'file':str(p),'sha256':sha(p),'bytes':p.stat().st_size},'overlaps':stats,'decoded_vs_existing_mp4_mae':diff})
   (D/'run.json').write_text(json.dumps(report,indent=2)+'\n');del out,z,raw[:],tiles,merged
  vae.decode=original;report['status']='complete';assert report['decoder_calls']==30
 except BaseException:report.update(status='failed_stop',error=traceback.format_exc());raise
 finally:
  report.update(seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated());(D/'run.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
