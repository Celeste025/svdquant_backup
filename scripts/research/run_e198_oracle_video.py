#!/usr/bin/env python3
"""Closed-loop H3 video comparison with exact-softmax and QK tile scoring."""
import argparse, importlib, json, math, os, sys, time, traceback, types
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/research'),str(ROOT/'scripts'),str(ROOT/'third_party/anemoi-code')]
import run_e162_proxy_new as previous
import run_e121_h3_split_generation as e121
from minimax_h3_svdquant_common import load_h3_pipeline,read_prompt
from diffsynth.utils.data.audio_video import write_video_audio
from anemoi import SparseConfig,QuantConfig,VisualLayout,anemoi_attention
from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout
from anemoi.layers.attention.mpa.routing import route_probability
import anemoi.layers.attention.mpa.executor as executor

RULES=('draftmap','qk_abs_mean','prob_mean','prob_rowmax_sum','qk_abs_top20_sum','prob_top20_sum','random')

@torch.inference_mode()
def score(q,k,prefix,grid,rule,seed):
    # q/k [1,S,H,D], official ragged order; scores computed from current trajectory.
    lay=materialize_ragged_2d_layout(q.device,frames=grid[0],height=grid[1],width=grid[2],logical_block=128,enable_anchors=False)
    counts=lay.counts; nb=counts.numel(); heads=q.size(2);block=128;dim=q.size(3)
    if rule=='random':
        gen=torch.Generator(device=q.device).manual_seed(seed)
        return torch.rand((1,heads,nb,nb),device=q.device,generator=gen)
    result=torch.empty((1,heads,nb,nb),device=q.device,dtype=torch.float32)
    valid=lay.slot_valid; keyvalid=torch.cat((torch.ones(prefix,device=q.device,dtype=torch.bool),valid))
    for begin in range(0,heads,4):
        end=min(begin+4,heads); h=end-begin
        qh=q[0,:,begin:end,:].transpose(0,1);kh=k[0,:,begin:end,:].transpose(0,1)
        qv=qh[:,lay.indices+prefix].float().view(h,nb,block,dim)
        keys=torch.cat((kh[:,:prefix].float(),kh[:,lay.indices+prefix].float()),dim=1)
        for row in range(nb):
            nq=int(counts[row]); x=torch.bmm(qv[:,row,:nq],keys.transpose(1,2))*dim**-.5
            x.masked_fill_(~keyvalid.view(1,1,-1),-torch.inf)
            if rule.startswith('prob_'):
                p=x.softmax(-1)[:,:,prefix:].view(h,nq,nb,block)
                if rule=='prob_mean':value=p.sum((1,3))/(nq*counts)
                elif rule=='prob_rowmax_sum':value=p.amax(-1).sum(1)
                else:
                    z=p.permute(0,2,1,3).reshape(h,nb,-1)
                    value=z.topk(min(20,z.size(-1)),dim=-1).values.sum(-1)
            else:
                z=x[:,:,prefix:].view(h,nq,nb,block).abs().masked_fill(~valid.view(1,1,nb,block),0)
                if rule=='qk_abs_mean':value=z.sum((1,3))/(nq*counts)
                else:
                    z=z.permute(0,2,1,3).reshape(h,nb,-1)
                    value=z.topk(min(20,z.size(-1)),dim=-1).values.sum(-1)
            result[0,begin:end,row]=value
    assert torch.isfinite(result).all()
    return result

class Router(previous.Router):
    def __init__(self,dit,arm):
        super().__init__(dit,arm)
        self.rule=ARGS.rule; self.skip=ARGS.skip
        self.sparse=SparseConfig(query_block_size=128,sparsity_ratio=self.skip,maxpool_weight=.2)
        nv=.6 if self.skip==.75 else .9
        self.quant=QuantConfig(nvfp4_ratio=nv,int8_ratio=1-nv,prefix_kv_precision='int8',prefix_query_precision='int8')
        self.manifest.update(rule=self.rule,skip=self.skip,all_main_steps_and_layers=True)
    def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
        log=self._active_log.get();block=self._active_main.get()
        expected=log.expected_refiner_cu if block is None else log.expected_cu
        assert softmax_scale==128**-.5 and q.shape==k.shape==v.shape==(expected[-1],56,128)
        log.remember_cu(cu_seqlens,expected,'refiner' if block is None else 'main')
        if block is None:
            log.refiner_helpers+=1
            return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
        n,total=expected[1:];tensors=[x[:n].unsqueeze(0).contiguous() for x in (q,k,v)]
        scores=None if self.rule=='draftmap' else score(tensors[0],tensors[1],self.prefix,self.grid,self.rule,198000+self.step*50+block)
        calls=[]
        def replace(probability,anchors,anchor_ids,*,anchor_count,sparsity_ratio,ratios):
            assert anchors is None and anchor_ids is None and anchor_count==0 and scores.shape==probability.shape
            result=route_probability(scores,None,anchor_count=0,prefix_blocks=0,sparsity_ratio=sparsity_ratio,
                nvfp4_ratio=ratios[0],fp8_ratio=ratios[1]+ratios[2],fp16_ratio=ratios[3])
            calls.append(1)
            return result
        kwargs=dict(layout=VisualLayout(self.grid,self.prefix),layer=block,sparse_config=self.sparse,quant_config=self.quant,calibration=previous.prior.prior.calibration(block))
        if scores is None:out0=anemoi_attention(*tensors,**kwargs)
        else:
            with patch.object(executor,'_route_sm120_probability',replace):out0=anemoi_attention(*tensors,**kwargs)
            assert len(calls)==1
        out=torch.empty_like(q);out[:n]=out0[0]
        if total>n:out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
        log.main_rows.append(dict(block=block,path='draft',capture=False,prefix_tokens=self.prefix,valid_length=n,padding_length=total-n))
        return out

@torch.inference_mode()
def main():
    global ARGS
    ap=argparse.ArgumentParser();ap.add_argument('--rule',choices=('bf16',*RULES),required=True);ap.add_argument('--skip',type=float,choices=(0.,.75),default=0.);ARGS=ap.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    importlib.import_module('diffsynth.core.attention.attention').ATTENTION_IMPLEMENTATION='torch'
    tag='bf16' if ARGS.rule=='bf16' else f'{ARGS.rule}_skip{int(ARGS.skip*100)}'
    experiment=os.environ.get('E198_EXPERIMENT','E198')
    report_path=ROOT/'results/research'/experiment/f'{tag}.json';video_path=ROOT/'results/local'/experiment/f'{tag}.mp4'
    assert not report_path.exists() and not video_path.exists()
    report_path.parent.mkdir(parents=True,exist_ok=True);video_path.parent.mkdir(parents=True,exist_ok=True)
    row=read_prompt(16,Path('/home/wjq/workspace/178866172854036')); report={'status':'running','rule':ARGS.rule,'skip':ARGS.skip,'prompt_id':16,'seed':int(row['seed']),
        'settings':{'height':480,'width':832,'frames':124,'steps':20,'cfg':1.,'flow_shift':12.,'audio_flow_shift':3.,'fps':24},
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'steps':[],'initial_latent_sha256':{},'video':str(video_path),'started':time.time()}
    def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
    save();router=None
    try:
        pipe=load_h3_pipeline(full=True,vram_limit_gib=18.);pipe.load_models_to_device(['dit']);pipe.dit.eval()
        report['linears']=e121.base.checked_linears(pipe.dit)
        if ARGS.rule!='bf16':
            previous.prior.MODE='proxy';router=Router(pipe.dit,'unused');old_forward=pipe.dit.forward
            def forward(_self,*args,**kwargs):
                contract=e121.base.attention_contract(kwargs);torch.cuda.synchronize();start=time.monotonic()
                with router.forward_context(**contract,diagnostics=False) as log: out=old_forward(*args,**kwargs)
                torch.cuda.synchronize();report['steps'].append({'step':len(report['steps']),'seconds':time.monotonic()-start,'main_calls':log.summary['main_helper_calls'],'mixed_calls':log.summary['sparse_calls']});save()
                print(f"{tag} step {len(report['steps'])}/20 {report['steps'][-1]['seconds']:.2f}s",flush=True);return out
            pipe.dit.forward=types.MethodType(forward,pipe.dit)
        old_step=pipe.step
        def step(_self,scheduler,latents,progress_id,noise_pred,**kwargs):
            kind='video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
            assert kind
            if progress_id==0:report['initial_latent_sha256'][kind]=e121.base.tensor_sha256(latents)
            return old_step(scheduler,latents,progress_id,noise_pred=noise_pred,**kwargs)
        pipe.step=types.MethodType(step,pipe)
        video,audio=pipe(prompt=row['prompt'],seed=int(row['seed']),height=480,width=832,num_frames=124,num_inference_steps=20,cfg_scale=1.,flow_shift=12.,audio_flow_shift=3.,rand_device='cpu',tiled=True)
        if router is not None:assert len(report['steps'])==20 and all(x['main_calls']==50 and x['mixed_calls']==50 for x in report['steps'])
        assert len(video)==124 and set(report['initial_latent_sha256'])=={'video','audio'}
        write_video_audio(video=video,audio=audio,output_path=str(video_path),fps=24,audio_sample_rate=pipe.audio_vae.sample_rate)
        report.update(status='complete',completed=time.time(),video_sha256=e121.base.sha256(video_path));save();print('E198_COMPLETE',tag,flush=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc(),failed=time.time());save();raise
    finally:
        if router is not None:router.close()
if __name__=='__main__':main()
