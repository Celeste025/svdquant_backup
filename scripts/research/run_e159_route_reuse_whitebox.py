#!/usr/bin/env python3
"""Four E150-aligned whitebox rollouts: proxy vs exact first-step reuse."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import torch
import run_e139_draftattention2 as prior
import run_e150_whitebox_bf16 as baseline
from anemoi import AttentionRouteCache, SparseConfig, QuantConfig, VisualLayout, anemoi_attention

ROOT=Path(__file__).resolve().parents[2]

class Router(prior.Router):
    dense_first_steps=1
    dense_first_layers=1
    def __init__(self,dit,arm):
        super().__init__(dit,'s70_fp4_20_int8_10')
        self.mode=MODE
        self.cache=AttentionRouteCache()
        self.sparse=SparseConfig(query_block_size=128,sparsity_ratio=0.,maxpool_weight=.2)
        self.quant=QuantConfig(nvfp4_ratio=.9,int8_ratio=.1,prefix_kv_precision='int8',prefix_query_precision='int8')
        self.manifest.update(mode=MODE,sparse_config=vars(self.sparse),quant_config=vars(self.quant),dense_first_steps=1,dense_first_layers=1,first_step_policy='dense INT8 except layer0 BF16; reuse additionally captures route',linears='BF16')
    @contextmanager
    def forward_context(self, **kwargs):
        with super().forward_context(**kwargs) as log:
            yield log
        if self.mode == 'reuse':
            assert len(self.cache) == 49
            self.manifest.update(cached_routes=len(self.cache), cached_route_bytes=self.cache.allocated_bytes)
    def bind_layout(self,module,args,kwargs):
        ids=kwargs['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long()
        pos=kwargs['img_position_ids'].detach().cpu().reshape(-1,3)[ids]
        self.grid=tuple(int(pos[:,i].unique().numel()) for i in range(3))
        assert self.grid[0]*self.grid[1]*self.grid[2]==ids.numel()
        self.prefix=int(ids[0]);assert torch.equal(ids,torch.arange(self.prefix,self.prefix+ids.numel()))
    def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
        log=self._active_log.get();block=self._active_main.get()
        expected=log.expected_refiner_cu if block is None else log.expected_cu
        assert softmax_scale == 128 ** -.5
        assert q.shape==k.shape==v.shape==(expected[-1],56,128)
        log.remember_cu(cu_seqlens,expected,'refiner' if block is None else 'main')
        if block is None:
            log.refiner_helpers+=1
            return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
        n,total=expected[1:]
        capture=self.mode=='reuse' and self.step==0 and block>=1
        dense=block<1
        if dense:out=self.original_helper(q,k,v,log.main_cpu_cu,softmax_scale)
        else:
            tensors=[t[:n].unsqueeze(0).contiguous() for t in (q,k,v)]
            extra={} if self.mode=='proxy' else dict(first_step=capture,route_cache=self.cache)
            quant=self.quant
            if self.mode=='proxy' and self.step==0:
                quant=QuantConfig(nvfp4_ratio=0.,int8_ratio=1.,prefix_kv_precision='int8',prefix_query_precision='int8')
            result=anemoi_attention(*tensors,layout=VisualLayout(self.grid,self.prefix),layer=block,sparse_config=self.sparse,quant_config=quant,calibration=prior.calibration(block),**extra)
            out=torch.empty_like(q);out[:n]=result[0]
            if total>n:out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
        log.main_rows.append(dict(block=block,path='dense' if dense or self.step==0 else 'draft',capture=capture,prefix_tokens=self.prefix,valid_length=n,padding_length=total-n))
        return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prompt-id',type=int,choices=(25,48,44,7),required=True);ap.add_argument('--mode',choices=('proxy','reuse'),required=True);a=ap.parse_args();MODE=a.mode
    bf=json.loads((ROOT/f'results/research/E150/p{a.prompt_id:03d}/bf16_denoise.json').read_text())
    assert bf['status']=='denoise_complete' and Path(bf['terminal_latents']).exists()
    baseline.REPORT=ROOT/'results/research/E159/aligned_v2'/MODE
    baseline.DATA=ROOT/'results/local/E159/aligned_v2'/MODE
    baseline.ARM=MODE
    baseline.e121.Router=Router
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    read_prompt=baseline.read_prompt
    baseline.read_prompt=lambda pid: read_prompt(pid,Path('/home/wjq/workspace/178866172854036'))
    baseline.denoise(a.prompt_id)
    result_path=baseline.REPORT/f'p{a.prompt_id:03d}/bf16_denoise.json'
    result=json.loads(result_path.read_text())
    assert result['initial_latent_sha256']==bf['initial_latent_sha256'], 'initial noise mismatch'
    x=torch.load(result['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].float()
    y=torch.load(bf['terminal_latents'],map_location='cpu',weights_only=True)['video_latents'].float()
    assert x.shape==y.shape
    result['comparison']={'bf16_reference':bf['terminal_latents'],'video_latent_nmse':float((x-y).double().square().sum()/y.double().square().sum()),'initial_noise_matches_bf16':True,'mode':MODE}
    result_path.write_text(json.dumps(result,indent=2)+'\n')
