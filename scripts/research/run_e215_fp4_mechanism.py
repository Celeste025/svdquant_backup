#!/usr/bin/env python3
"""Decompose sparse/mixed error and intervene on Q/K/V scales on BF16 trajectory."""
import argparse,importlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'scripts/research'),str(ROOT/'third_party/anemoi-code')]
from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
from diffsynth.models import minimax_h3_dit_comfy as comfy
from anemoi import anemoi_attention,SparseConfig,QuantConfig,VisualLayout,NVFP4Calibration
OUT=ROOT/'results/research/E215';SCALE=ROOT/'third_party/anemoi-code/anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json'
class Done(Exception):pass
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prompt',type=int,required=True);args=ap.parse_args();pid=args.prompt
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch';row=read_prompt(pid);pipe=load_h3_pipeline(full=True,vram_limit_gib=18.);pipe.load_models_to_device(['dit']);pipe.dit.eval()
 state={'step':-1,'layer':None};hooks=[];grid=(37,15,26);scales=json.loads(SCALE.read_text())['scales'];results=[]
 expected={(r['step'],r['layer']):r for r in map(json.loads,(ROOT/f'results/research/E214/p{pid:03d}/cells.jsonl').open()) if r['step']==14 and r['layer'] in (25,49)}
 def bind(_m,_a,kw):
  state['step']+=1;ids=kw['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long();state['prefix']=int(ids[0]);assert ids.numel()==37*15*26 and torch.equal(ids,torch.arange(state['prefix'],state['prefix']+ids.numel()))
 hooks.append(pipe.dit.register_forward_pre_hook(bind,with_kwargs=True))
 for layer in (25,49):
  def enter(*_a,layer=layer):state['layer']=layer
  def leave(*_a):state['layer']=None
  hooks.extend((pipe.dit.blocks[layer].attn.register_forward_pre_hook(enter),pipe.dit.blocks[layer].attn.register_forward_hook(leave)))
 original=comfy._sdpa_varlen_attention
 def intercept(q,k,v,cu_seqlens,softmax_scale):
  layer=state['layer'];step=state['step']
  if layer not in (25,49) or step!=14:return original(q,k,v,cu_seqlens,softmax_scale)
  n=cu_seqlens.detach().cpu().tolist()[1];prefix=state['prefix'];assert n==prefix+37*15*26 and q.shape[1:]==(56,128)
  ref=original(q,k,v,cu_seqlens,softmax_scale);target=ref[prefix:n].float();inputs=[x[:n].unsqueeze(0).contiguous() for x in (q,k,v)];cal=[float(scales[x][layer]) for x in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')];den=target.square().sum((0,2),dtype=torch.float64)
  def run(skip,nv,i8,fp16,calibration):
   pred=anemoi_attention(*inputs,layout=VisualLayout(grid,prefix),layer=layer,sparse_config=SparseConfig(query_block_size=64,sparsity_ratio=skip,maxpool_weight=.2),quant_config=QuantConfig(nvfp4_ratio=nv,int8_ratio=i8,fp16_ratio=fp16,prefix_kv_precision='int8',prefix_query_precision='int8' if i8 else 'fp16'),calibration=NVFP4Calibration(*calibration) if nv else None)[0,prefix:n].float()
   assert torch.isfinite(pred).all();return pred
  def metrics(pred,sparse_ref=None):
   e=pred-target;num=e.square().sum((0,2),dtype=torch.float64);result={'nmse':float(num.sum()/den.sum()),'head_nmse':(num/den).cpu().tolist()}
   if sparse_ref is not None:
    es=sparse_ref-target;eq=pred-sparse_ref;ss=es.square().sum((0,2),dtype=torch.float64);sq=eq.square().sum((0,2),dtype=torch.float64);cross=2*(es.double()*eq.double()).sum((0,2));result.update(sparse_nmse=float(ss.sum()/den.sum()),quant_nmse=float(sq.sum()/den.sum()),cross_nmse=float(cross.sum()/den.sum()),head_sparse_nmse=(ss/den).cpu().tolist(),head_quant_nmse=(sq/den).cpu().tolist(),head_cross_nmse=(cross/den).cpu().tolist())
   return result
  # Per-head maxima and RMS for actual K, plus occupancy of official range.
  kv=k[prefix:n].float();khead={'rms':kv.square().mean((0,2)).sqrt().cpu().tolist(),'absmax':kv.abs().amax((0,2)).cpu().tolist(),'fraction_over_half_range':(kv.abs()>1344*cal[1]).float().mean((0,2)).cpu().tolist()};del kv
  data={'prompt_id':pid,'step':step,'layer':layer,'calibration':cal,'k_video_stats':khead,'reference_energy_by_head':den.cpu().tolist(),'arms':{},'scale_interventions':{},'same_support_all_int8':{}}
  for index,(skip,nv,i8) in enumerate(((.8,.5,.5),(.5,.8,.2),(0.,.9,.1))):
   dense_support=run(skip,0,0,1,cal);mixed=run(skip,nv,i8,0,cal);m=metrics(mixed,dense_support)
   prev=expected[(step,layer)]['arms'][index];diff=max(abs(a-b) for a,b in zip(m['head_nmse'],prev['nmse_by_head']));assert diff<1e-6,(pid,layer,index,diff)
   m['replay_max_head_abs_nmse_diff']=diff;data['arms'][str(skip)]=m
   data['same_support_all_int8'][str(skip)]=metrics(run(skip,0,1,0,cal))
   del mixed,dense_support
  for comp in range(3):
   for factor in (.5,2.):
    c=cal.copy();c[comp]*=factor
    for skip,nv,i8 in ((.8,.5,.5),(0.,.9,.1)):
     key=f'{"qkv"[comp]}_x{factor:g}_skip{skip:g}';data['scale_interventions'][key]=metrics(run(skip,nv,i8,0,c))
  results.append(data);(OUT/f'p{pid:03d}_step14_layer{layer}.json').write_text(json.dumps(data,indent=2)+'\n');print('DONE_LAYER',pid,layer,flush=True)
  if layer==49:raise Done()
  return ref
 comfy._sdpa_varlen_attention=intercept
 try:
  try:pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
  except Done:pass
  else:raise RuntimeError('target not reached')
 finally:
  comfy._sdpa_varlen_attention=original
  for h in hooks:h.remove()
 assert len(results)==2;print('COMPLETE',pid,flush=True)
if __name__=='__main__':main()
