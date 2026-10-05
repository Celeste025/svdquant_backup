"""E038: E016 H3 routing with E034 coarse16 centers and the frozen E033 consumer.

No model/projection/normalization/RoPE changes. Only the valid main segment uses
FP4; original refiner and padding attention stay BF16. Geometry follows actual N.
"""
from __future__ import annotations
import importlib
import torch
import torch.nn.functional as F
from h3_native_fp4_attention import H3AttentionRouter, HEADS, HEAD_DIM, EXPECTED_SCALE, _require, _source

ARMS = ('bf16', 'global_mean', 'coarse16')


def geometry(n, heads=HEADS):
    groups = (int(n)+127)//128
    _require(groups >= 16, 'coarse16 requires at least 16 nonempty query tiles')
    boundaries = [j*groups//16 for j in range(17)]
    ids = torch.empty(groups, dtype=torch.int32)
    for j, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ids[lo:hi] = j
    return dict(valid_length=int(n), padded_length=groups*128, groups=groups,
                boundaries=boundaries), ids[None, None].expand(1, heads, groups).contiguous()


def centered_operands(q, k, v, geo, ids):
    """BF16 HND operands; Q padding enters means, valid K only enters K mean."""
    h, n, d = q.shape
    np, g = geo['padded_length'], geo['groups']
    _require(k.shape == v.shape == q.shape and n == geo['valid_length'], 'QKV/geometry mismatch')
    qpad = F.pad(q, (0, 0, 0, np-n))
    centers = torch.stack([qpad[:, 128*lo:128*hi].mean(1)
        for lo, hi in zip(geo['boundaries'][:-1], geo['boundaries'][1:])], dim=1)
    selected = centers.gather(1, ids[0].long().unsqueeze(-1).expand(-1, -1, d))
    qc = (qpad.reshape(h, g, 128, d)-selected.unsqueeze(2)).reshape(h, np, d).contiguous()
    mean = k.mean(-2, keepdim=True)
    kc = F.pad(k-mean, (0, 0, 0, np-n))
    vp = F.pad(v, (0, 0, 0, np-n))
    return qc, kc, vp, centers


class H3CenterRouter(H3AttentionRouter):
    def __init__(self, dit, mode, **kwargs):
        _require(mode in ARMS, f'Unknown E038 mode: {mode}')
        super().__init__(dit, 'global_mean' if mode == 'coarse16' else mode, **kwargs)
        self.mode = mode
        self._geometry = {}
        self.private = importlib.import_module('codebook_attention_sm120') if mode == 'coarse16' else None
        self.manifest.update(mode=mode, official_per_block_mean=None if mode == 'coarse16'
            else (None if mode == 'bf16' else False),
            center_rule='floor(j*ceil(N/128)/16), direct BF16 padded Q means; BF16 valid K mean; FP32 C@Kc.T',
            coarse_table_rows=16 if mode == 'coarse16' else None)
        self.manifest['sources']['e038_router'] = _source(__file__)
        if self.private is not None:
            self.manifest['sources']['private_consumer'] = _source(self.private.__file__)

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        if self.mode != 'coarse16' or self._active_main.get() is None:
            return super()._dispatch(q, k, v, cu_seqlens, softmax_scale)
        log = self._active_log.get()
        _require(log is not None, 'Attention outside forward_context')
        n, total = log.expected_cu[1:]
        _require(q.shape == k.shape == v.shape == (total, HEADS, HEAD_DIM), 'Actual H3 QKV shape changed')
        _require(all(t.dtype == torch.bfloat16 and t.device == q.device for t in (q, k, v)), 'QKV dtype/device changed')
        _require(self._cpu_fixture or q.device.type == 'cuda', 'Production requires CUDA')
        _require(softmax_scale == EXPECTED_SCALE, 'H3 scale changed')
        log.remember_cu(cu_seqlens, log.expected_cu, 'main')
        cache_key = (n, str(q.device))
        if cache_key not in self._geometry:
            geo, ids = geometry(n)
            self._geometry[cache_key] = geo, ids.to(q.device)
        geo, ids = self._geometry[cache_key]
        np = geo['padded_length']
        dense = tuple(t[:n].transpose(0, 1).contiguous() for t in (q, k, v))
        qc, kc, vp, centers = centered_operands(*dense, geo, ids)
        module = self.official.get_nvfp4_attention_sm120_module()
        q4 = torch.empty((1, HEADS, np, HEAD_DIM//2), dtype=torch.uint8, device=q.device)
        qs = torch.empty((1, HEADS, np, HEAD_DIM//16), dtype=torch.float8_e4m3fn, device=q.device)
        k4, ks = torch.empty_like(q4), torch.empty_like(qs)
        vt = torch.empty((1, HEADS, HEAD_DIM, np//2), dtype=torch.uint8, device=q.device)
        vs = torch.empty((1, HEADS, HEAD_DIM, np//16), dtype=torch.float8_e4m3fn, device=q.device)
        module.scaled_fp4_quant(qc.unsqueeze(0), q4, qs, 1)
        module.scaled_fp4_quant_permute(kc.unsqueeze(0), k4, ks, 1)
        module.scaled_fp4_quant_trans(vp.unsqueeze(0), vt, vs, 1)
        # Match E034 operands: both BF16 inputs converted to FP32 before matmul.
        table = (centers.float() @ kc.transpose(-2, -1).float()).unsqueeze(0).contiguous()
        del dense, qc, kc, vp, centers
        result = self.private.codebook_fwd(q4, k4, vt, qs, ks, vs, table, ids,
            sm_scale=softmax_scale, unpadded_k_len=n)
        _require(result.shape == (1, HEADS, np, HEAD_DIM) and result.dtype == torch.bfloat16,
                 'Private output ABI changed')
        log.fp4_calls += 1
        log.main_rows.append(dict(block=self._active_main.get(), mode=self.mode,
            input_shape=list(q.shape), valid_length=n, padding_length=total-n,
            official_output_shape=list(result.shape), correction_shape=list(table.shape),
            correction_bytes=table.numel()*table.element_size(), unpadded_k_len=n,
            boundaries=geo['boundaries'], q_pack=1, k_pack=1, v_pack=1,
            coarse_means=16, correction_gemm=1, private_attention=1,
            causal=False, return_lse=False, softmax_scale=softmax_scale))
        out = torch.empty_like(q)
        out[:n] = result[0, :, :n].transpose(0, 1)
        out[n:] = self.original_helper(q[n:], k[n:], v[n:], log.padding_cpu_cu, softmax_scale)
        log.original_bf16_segments += 1
        log.finite(out)
        return out


def install_h3_center_attention(dit, mode):
    return H3CenterRouter(dit, mode)


def cpu_geometry_check(lengths):
    """Small geometry/padding check, no imported GPU module or JIT invocation."""
    rows = []
    for n in sorted(set([2048, 2307, *map(int, lengths)])):
        geo, ids = geometry(n, heads=2)
        assert ids.shape == (1, 2, geo['groups']) and ids.min() == 0 and ids.max() == 15
        assert sum(b-a for a, b in zip(geo['boundaries'][:-1], geo['boundaries'][1:])) == geo['groups']
        rows.append(geo)
    geo, ids = geometry(2307, heads=2)
    q = torch.ones((2, 2307, 128), dtype=torch.bfloat16)
    k, v = 2*q, 3*q
    qc, kc, vp, c = centered_operands(q, k, v, geo, ids)
    begin = 128*geo['boundaries'][-2]
    expected_tail = torch.tensor((2307-begin)/(geo['padded_length']-begin), dtype=torch.bfloat16)
    assert torch.equal(c[:, -1], torch.full_like(c[:, -1], expected_tail.item()))
    assert torch.count_nonzero(kc) == 0 and torch.count_nonzero(vp[:, 2307:]) == 0
    assert torch.equal(qc[:, 2307:], -c[:, -1:].expand(-1, geo['padded_length']-2307, -1))
    assert not torch.cuda.is_initialized()
    return dict(status='complete', geometries=rows, q_padding_in_mean=True,
                k_padding_excluded_then_zero=True, cuda_initialized=False)
