#!/usr/bin/env python3
"""Full H3 video: capture first-step QK-max or INT8 probability route; reuse."""
import os
os.environ['E198_EXPERIMENT']='E201'
import torch
import run_e198_oracle_video as base
from anemoi import AttentionRouteCache,VisualLayout,anemoi_attention

class ReuseRouter(base.Router):
    def __init__(self,dit,arm):
        super().__init__(dit,arm)
        self.cache=AttentionRouteCache(capture_mode='softmax_mean' if self.rule=='reuse_prob' else 'qk_max')
        self.manifest.update(capture_mode=self.cache.capture_mode,first_step_policy='dense INT8 route capture; all later steps reuse same layer route')
    def _dispatch(self,q,k,v,cu_seqlens,softmax_scale):
        log=self._active_log.get();block=self._active_main.get()
        expected=log.expected_refiner_cu if block is None else log.expected_cu
        assert softmax_scale==128**-.5 and q.shape==k.shape==v.shape==(expected[-1],56,128)
        log.remember_cu(cu_seqlens,expected,'refiner' if block is None else 'main')
        if block is None:
            log.refiner_helpers+=1
            return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
        n,total=expected[1:];tensors=[x[:n].unsqueeze(0).contiguous() for x in (q,k,v)]
        result=anemoi_attention(*tensors,layout=VisualLayout(self.grid,self.prefix),layer=block,sparse_config=self.sparse,quant_config=self.quant,calibration=base.previous.prior.prior.calibration(block),first_step=self.step==0,route_cache=self.cache)
        out=torch.empty_like(q);out[:n]=result[0]
        if total>n:out[n:]=self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
        log.main_rows.append(dict(block=block,path='draft',capture=self.step==0,prefix_tokens=self.prefix,valid_length=n,padding_length=total-n))
        return out

if __name__=='__main__':
    base.RULES=('reuse_qk','reuse_prob')
    base.Router=ReuseRouter
    base.main()
