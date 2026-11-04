#!/usr/bin/env python3
"""Paired attention error across NVFP4 budgets on a BF16 H3 trajectory."""
import argparse,hashlib,importlib,json,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy
from anemoi import anemoi_attention,SparseConfig,QuantConfig,VisualLayout,NVFP4Calibration
SCALE=ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json'
OUT=ROOT/'results/research/E214'
class Done(Exception):pass
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prompt',type=int,required=True);args=ap.parse_args();pid=args.prompt
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
 row=read_prompt(pid);folder=OUT/f'p{pid:03d}';folder.mkdir(parents=True,exist_ok=True);path=folder/'cells.jsonl';assert not path.exists()
 scales=json.loads(SCALE.read_text())['scales'];cals=[NVFP4Calibration(*(float(scales[k][i]) for k in ('q_tensor_scale','k_tensor_scale','v_tensor_scale'))) for i in range(50)]
 specs=[(.8,.5,.5),(.5,.8,.2),(0.,.9,.1)]
 sparse=[SparseConfig(query_block_size=64,sparsity_ratio=s,maxpool_weight=.2) for s,_,_ in specs]
 quant=[QuantConfig(nvfp4_ratio=n,int8_ratio=i,fp16_ratio=0.,prefix_kv_precision='int8',prefix_query_precision='int8') for _,n,i in specs]
 report={'status':'running','prompt_id':pid,'seed':int(row['seed']),'settings':{'height':480,'width':832,'frames':124,'steps':20,'cfg':1.,'flow_shift':12.,'audio_flow_shift':3.,'rand_device':'cpu','layers':50,'heads':56,'tile':64,'maxpool_weight':.2},'arms_skip_nvfp4_int8':[[.8,.1,.1],[.5,.4,.1],[0.,.9,.1]],'calibration_sha256':hashlib.sha256(SCALE.read_bytes()).hexdigest(),'teacher':'BF16 dense, original attention output returned unchanged','started_unix':time.time()}
 (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 pipe=load_h3_pipeline(full=True,vram_limit_gib=18.);pipe.load_models_to_device(['dit']);pipe.dit.eval();state={'step':-1,'layer':None,'count':0};handles=[];grid=(37,15,26)
 def bind(_m,_a,kw):
  state['step']+=1;ids=kw['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long();state['prefix']=int(ids[0]);assert ids.numel()==37*15*26 and torch.equal(ids,torch.arange(state['prefix'],state['prefix']+ids.numel()))
 handles.append(pipe.dit.register_forward_pre_hook(bind,with_kwargs=True))
 for layer in range(50):
  def enter(*_args,layer=layer):state['layer']=layer
  def leave(*_args):state['layer']=None
  handles.extend([pipe.dit.blocks[layer].attn.register_forward_pre_hook(enter),pipe.dit.blocks[layer].attn.register_forward_hook(leave)])
 original=comfy._sdpa_varlen_attention
 def intercept(q,k,v,cu_seqlens,softmax_scale):
  layer=state['layer']
  if layer is None:return original(q,k,v,cu_seqlens,softmax_scale)
  step=state['step'];assert state['count']==step*50+layer
  bounds=cu_seqlens.detach().cpu().tolist();n=bounds[1];prefix=state['prefix'];assert n==prefix+37*15*26 and q.shape==k.shape==v.shape and q.shape[1:]==(56,128) and softmax_scale==128**-.5
  ref=original(q,k,v,cu_seqlens,softmax_scale);assert ref.shape==q.shape
  inputs=[x[:n].unsqueeze(0).contiguous() for x in (q,k,v)];target=ref[prefix:n].float();den=target.square().sum((0,2),dtype=torch.float64);assert torch.isfinite(target).all() and (den>0).all()
  arms=[]
  for sc,qc in zip(sparse,quant):
   pred=anemoi_attention(*inputs,layout=VisualLayout(grid,prefix),layer=layer,sparse_config=sc,quant_config=qc,calibration=cals[layer])[0,prefix:n].float()
   assert torch.isfinite(pred).all();numer=(pred-target).square().sum((0,2),dtype=torch.float64);nmse=numer/den
   arms.append({'nmse_by_head':nmse.cpu().tolist(),'error_energy_by_head':numer.cpu().tolist(),'nmse_overall':float(numer.sum()/den.sum())})
  record={'prompt_id':pid,'step':step,'layer':layer,'prefix_tokens':prefix,'video_grid':grid,'reference_energy_by_head':den.cpu().tolist(),'arms':arms}
  with path.open('a') as f:f.write(json.dumps(record,separators=(',',':'))+'\n');f.flush()
  state['count']+=1
  if layer==49:print('STEP',pid,step,'cells',state['count'],flush=True)
  if step==19 and layer==49:raise Done()
  return ref
 comfy._sdpa_varlen_attention=intercept
 try:
  try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
  except Done:pass
  else:raise RuntimeError('last cell not reached')
  assert state['count']==1000;report.update(status='complete',cells=state['count'],completed_unix=time.time())
 except BaseException:
  report.update(status='failed',cells=state['count'],error=traceback.format_exc());raise
 finally:
  comfy._sdpa_varlen_attention=original
  for h in handles:h.remove()
  (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',pid,flush=True)
if __name__=='__main__':main()
