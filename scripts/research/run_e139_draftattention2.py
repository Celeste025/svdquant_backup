#!/usr/bin/env python3
"""Official Anemoi/DraftAttention2 in the existing paired H3 pipeline."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'third_party/anemoi-code'
sys.path[:0] = [str(SOURCE), str(ROOT/'scripts'), str(ROOT/'scripts/research')]
from anemoi import anemoi_attention, SparseConfig, QuantConfig, VisualLayout, NVFP4Calibration
import run_e121_h3_split_generation as e121
from h3_native_fp4_attention import AttentionCallLog

REPORT = ROOT/'results/research/E139'
DATA = Path('/data1/models/svdquant-wjq/research/20261031/E139_draftattention2')
GRID = (37,18,32)


def calibration(layer):
    path = SOURCE/'anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json'
    assert e121.base.sha256(path) == 'ef5413c53e459b8210c8fe3054a5462310f050ebc8a63e5d20a397a90627d455'
    data = json.loads(path.read_text())['scales']
    return NVFP4Calibration(*(float(data[name][layer]) for name in
                             ['q_tensor_scale','k_tensor_scale','v_tensor_scale']))


class Router(e121.Router):
    dense_first_steps = 10
    dense_first_layers = 2
    def __init__(self, dit, arm):
        super().__init__(dit, 'bf16')
        self.arm = arm
        self.step = -1
        self.prefix = None
        self.sparse = SparseConfig(query_block_size=64, sparsity_ratio=.70, maxpool_weight=.2)
        mixed = arm == 's70_fp4_20_int8_10'
        self.quant = QuantConfig(nvfp4_ratio=2/3 if mixed else 0.,
                                int8_ratio=1/3 if mixed else 1., fp16_ratio=0.,
                                prefix_kv_precision='int8', prefix_query_precision='int8')
        self.manifest.update(arm=arm, code_commit='97a1c8b14f26f387214c8cfb8fc7e890390b93b4',
                             sparse_config=vars(self.sparse), quant_config=vars(self.quant),
                             dense_first_steps=self.dense_first_steps,
                             dense_first_layers=self.dense_first_layers, video_grid=list(GRID),
                             prefix_policy='official dense INT8', linears='BF16')
        self.layout_hook = dit.register_forward_pre_hook(self.bind_layout, with_kwargs=True)

    def bind_layout(self, module, args, kwargs):
        ids = kwargs['img_pos_info']['position_ids'].detach().cpu().reshape(-1).long()
        assert ids.numel() == 37*18*32, (ids.shape, 'unexpected video tokens')
        start = int(ids[0])
        assert torch.equal(ids, torch.arange(start, start+ids.numel())), 'video is not a contiguous suffix'
        # nn.Module pre-hooks run before the observed forward opens the scoped
        # attention context. Read the same sequence contract from kwargs here.
        n = int(kwargs['packed_seq_params']['cu_seqlens_q'][1].detach().cpu())
        assert start+ids.numel() == n, (start, ids.numel(), n)
        self.prefix = start

    @contextmanager
    def forward_context(self, *, expected_cu, expected_refiner_cu, diagnostics=True):
        self.step += 1
        log = AttentionCallLog('bf16', expected_cu, expected_refiner_cu, diagnostics)
        token = self._active_log.set(log)
        try:
            yield log
            assert [r['block'] for r in log.main_rows] == list(range(50))
            assert log.refiner_helpers == 2
            expected_sparse = (0 if self.arm == 'bf16' or self.step < self.dense_first_steps
                               else 50-self.dense_first_layers)
            assert sum(r['path']=='draft' for r in log.main_rows) == expected_sparse
            for cu, expected, _ in log.cu_references.values():
                assert tuple(cu.cpu().tolist()) == expected
            log.summary = {'main_helper_calls': 50, 'refiner_helper_calls': 2,
                           'expected_cu': list(expected_cu), 'main_rows': log.main_rows,
                           'sparse_calls': sum(r['path']=='draft' for r in log.main_rows)}
        finally:
            self._active_log.reset(token)

    def _dispatch(self, q,k,v,cu_seqlens,softmax_scale):
        log = self._active_log.get()
        block = self._active_main.get()
        expected = log.expected_refiner_cu if block is None else log.expected_cu
        assert q.shape == k.shape == v.shape == (expected[-1],56,128)
        assert softmax_scale == 128**-.5
        log.remember_cu(cu_seqlens, expected, 'refiner' if block is None else 'main')
        if block is None:
            log.refiner_helpers += 1
            return self.original_helper(q,k,v,cu_seqlens,softmax_scale)
        n,total = expected[1:]
        dense = (self.arm == 'bf16' or self.step < self.dense_first_steps
                 or block < self.dense_first_layers)
        if dense:
            out = self.original_helper(q,k,v,log.main_cpu_cu,softmax_scale)
        else:
            assert self.prefix is not None
            tensors = [t[:n].unsqueeze(0).contiguous() for t in (q,k,v)]
            result = anemoi_attention(*tensors, layout=VisualLayout(GRID,self.prefix), layer=block,
                                     sparse_config=self.sparse, quant_config=self.quant,
                                     calibration=calibration(block) if self.quant.nvfp4_ratio else None)
            assert result.shape == tensors[0].shape and result.dtype == q.dtype
            out = torch.empty_like(q)
            out[:n] = result[0]
            if total > n:
                out[n:] = self.original_helper(q[n:],k[n:],v[n:],log.padding_cpu_cu,softmax_scale)
        log.main_rows.append({'block':block, 'path':'dense' if dense else 'draft',
                              'prefix_tokens':self.prefix,'valid_length':n,'padding_length':total-n})
        return out

    def close(self):
        self.layout_hook.remove()
        super().close()


@torch.inference_mode()
def smoke():
    torch.manual_seed(139)
    q,k,v = [torch.randn(1,273,2,128,device='cuda',dtype=torch.bfloat16)*.3 for _ in range(3)]
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    rows=[]
    for mode in ['fp16_dense','int8_sparse','mixed_sparse']:
        sparse=SparseConfig(sparsity_ratio=0. if mode=='fp16_dense' else .70)
        quant=QuantConfig(nvfp4_ratio=2/3 if mode=='mixed_sparse' else 0.,
                          int8_ratio=1. if mode=='int8_sparse' else 1/3 if mode=='mixed_sparse' else 0.,
                          fp16_ratio=1. if mode=='fp16_dense' else 0.,
                          prefix_kv_precision='fp16',prefix_query_precision='fp16')
        out=anemoi_attention(q,k,v,layout=VisualLayout((1,16,16),17),layer=0,
                             sparse_config=sparse,quant_config=quant)
        assert torch.isfinite(out).all()
        nmse=float((out.float()-reference.float()).square().sum()/reference.float().square().sum())
        if mode=='fp16_dense':
            assert nmse < 1e-4, nmse
        rows.append({'mode':mode,'nmse':nmse})
    e121.base.save_json(REPORT/'operator_smoke.json',{'rows':rows,'shape':list(q.shape)})
    print(rows,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--case',choices=['race','clap0'])
    ap.add_argument('--arm',choices=['bf16','s70_int8_30','s70_fp4_20_int8_10'])
    ap.add_argument('--phase',choices=['smoke','denoise','decode'],required=True)
    args=ap.parse_args()
    if args.phase=='smoke':
        return smoke()
    e121.base.REPORT=REPORT
    e121.base.DATA=DATA
    e121.base.PLAN=REPORT/'experiment_plan.md'
    e121.base.Router=Router
    torch.set_num_threads(6)
    if args.phase=='denoise':
        e121.base.denoise(args.case,args.arm)
    else:
        e121.base.decode(args.case,args.arm)


if __name__=='__main__':
    main()
