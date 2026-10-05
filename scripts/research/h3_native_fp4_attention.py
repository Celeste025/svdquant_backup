"""E016 routing adapter for official SM120 NVFP4 attention.

QKV projection, QK normalization, RoPE, and output projection remain in the
original bound H3 forward. Only the first segment of 50 main attention helpers
uses the official quantize/fwd pair. Refiner and padding use the original helper.
There is no fallback. CPU fixtures below test routing, not GPU arithmetic.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import json
from pathlib import Path

import torch

MODES = ('bf16', 'block_mean', 'global_mean')
HEADS, HEAD_DIM = 56, 128
EXPECTED_SCALE = HEAD_DIM ** -0.5


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _source(path):
    path = Path(path).resolve()
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


class AttentionCallLog:
    def __init__(self, mode, expected_cu, expected_refiner_cu, diagnostics):
        self.mode = mode
        self.expected_cu = tuple(int(v) for v in expected_cu)
        self.expected_refiner_cu = tuple(int(v) for v in expected_refiner_cu)
        _require(len(self.expected_cu) == 3 and self.expected_cu[0] == 0
                 and 0 < self.expected_cu[1] < self.expected_cu[2], 'Expected nonempty main and padding segments')
        _require(len(self.expected_refiner_cu) == 3 and self.expected_refiner_cu[0] == 0
                 and self.expected_refiner_cu[1] == self.expected_refiner_cu[2] > 0,
                 'Expected one nonempty refiner segment')
        # Shared CPU metadata preparation for every main-attention mode. The
        # BF16 baseline also avoids per-layer GPU cu.tolist() synchronization.
        self.main_cpu_cu = torch.tensor(self.expected_cu, dtype=torch.int32)
        self.padding_cpu_cu = torch.tensor([0, self.expected_cu[2]-self.expected_cu[1]], dtype=torch.int32)
        self.diagnostics = bool(diagnostics)
        self.main_rows = []
        self.refiner_helpers = 0
        self.fp4_calls = 0
        self.original_bf16_segments = 0
        self.cu_references = {}
        self.finite_flags = []
        self.summary = None

    def remember_cu(self, cu, expected, kind):
        _require(torch.is_tensor(cu) and cu.dtype == torch.int32 and tuple(cu.shape) == (3,),
                 'Original cu_seqlens dtype/shape changed')
        # Retain only tiny metadata tensors. No .tolist()/bool(CUDA) occurs in a
        # main FP4 helper. Full forward timing must end before context exit.
        key = (str(cu.device), cu.data_ptr(), tuple(expected))
        if key not in self.cu_references:
            self.cu_references[key] = (cu, tuple(expected), kind)

    def finite(self, tensor):
        if self.diagnostics:
            # Only scalar flags survive this call; large activations are not
            # retained. GPU reductions are present in diagnostic runs only.
            self.finite_flags.append(torch.isfinite(tensor).all())

    def finish(self):
        low = self.mode != 'bf16'
        cu_rows = []
        for cu, expected, kind in self.cu_references.values():
            actual = cu.detach().cpu().tolist()
            cu_rows.append({'kind': kind, 'actual': actual, 'expected': list(expected),
                            'exact': tuple(actual) == expected})
        finite_ok = None
        if self.diagnostics:
            _require(bool(self.finite_flags), 'No diagnostic finite flags collected')
            finite_ok = bool(torch.stack(self.finite_flags).all().cpu())
        self.summary = {
            'mode': self.mode, 'main_helper_calls': len(self.main_rows),
            'refiner_helper_calls': self.refiner_helpers, 'fp4_calls': self.fp4_calls,
            'original_bf16_segments': self.original_bf16_segments,
            'expected_cu': list(self.expected_cu), 'expected_refiner_cu': list(self.expected_refiner_cu),
            'actual_cu_checks': cu_rows, 'diagnostics': self.diagnostics,
            'finite_flag_count': len(self.finite_flags), 'finite': finite_ok,
            'main_rows': self.main_rows,
            'timing_boundary': 'Context exit checks tiny cu metadata and optional finite flags after the caller end synchronization',
            'metadata_policy': 'All main modes use the same prebound CPU boundaries and deferred actual-cu verification; refiner uses original boundaries/helper in every mode.',
        }
        _require([r['block'] for r in self.main_rows] == list(range(50)), 'Expected main blocks 0..49 exactly once')
        _require(self.refiner_helpers == 2, 'Expected exactly two original refiner helpers')
        _require(self.fp4_calls == (50 if low else 0), 'Unexpected official FP4 attention call count')
        _require(self.original_bf16_segments == (52 if low else 102), 'Unexpected original BF16 segment count')
        _require(all(r['exact'] for r in cu_rows), 'Actual attention boundaries differ from bound case')
        _require(finite_ok is not False, 'Nonfinite QKV/correction/attention output')


class H3AttentionRouter:
    def __init__(self, dit, mode, *, _comfy=None, _official=None, _cpu_fixture=False):
        _require(mode in MODES, f'Unknown attention mode {mode}')
        self.mode = mode
        self.comfy = _comfy or importlib.import_module('diffsynth.models.minimax_h3_dit_comfy')
        self.official = _official
        if self.official is None and mode != 'bf16':
            self.official = importlib.import_module('flashinfer.nvfp4_attention_sm120')
        self.original_helper = self.comfy._sdpa_varlen_attention
        _require(not isinstance(getattr(self.original_helper, '__self__', None), H3AttentionRouter),
                 'Another E016 router is already installed')
        _require(len(dit.blocks) == 50 and len(dit.token_refiner.blocks) == 2, 'Unexpected H3 attention coverage')
        self._cpu_fixture = _cpu_fixture
        self._active_main = ContextVar(f'e016_main_{id(self)}', default=None)
        self._active_log = ContextVar(f'e016_log_{id(self)}', default=None)
        self.original_forwards = []
        self.closed = False
        for index, block in enumerate(dit.blocks):
            attention = block.attn
            _require(attention.num_heads == HEADS and attention.head_dim == HEAD_DIM
                     and attention.softmax_scale == EXPECTED_SCALE, 'Unexpected H3 head/scale contract')
            original = attention.forward
            self.original_forwards.append((attention, original))
            attention.forward = self._scoped_forward(original, index)
        self.comfy._sdpa_varlen_attention = self._dispatch
        sources = {'adapter': _source(__file__)}
        if not _cpu_fixture:
            sources['comfy'] = _source(self.comfy.__file__)
            base_model = importlib.import_module('diffsynth.models.minimax_h3_dit')
            sources['original_helper'] = _source(base_model.__file__)
            if self.official is not None:
                sources['official_attention'] = _source(self.official.__file__)
        self.manifest = {
            'mode': mode, 'sources': sources, 'main_blocks': list(range(50)), 'refiner_blocks': 2,
            'official_per_block_mean': None if mode == 'bf16' else mode == 'block_mean',
            'reference_operations': 'Original bound forward preserves QKV, QK RMSNorm, RoPE and out projection',
            'global_mean_semantics': 'Official K centering before padding and Q mean after padding; not no-centering',
            'metadata_policy': 'BF16 and both FP4 modes share prebound CPU main boundaries and deferred actual-cu verification; refiner helper remains unchanged for all modes.',
            'cpu_fixture_only': _cpu_fixture,
        }

    def _scoped_forward(self, original, index):
        def scoped(*args, **kwargs):
            _require(self._active_log.get() is not None, 'Forward requires router.forward_context')
            _require(self._active_main.get() is None, 'Unexpected nested main attention')
            token = self._active_main.set(index)
            try:
                return original(*args, **kwargs)
            finally:
                self._active_main.reset(token)
        return scoped

    @contextmanager
    def forward_context(self, *, expected_cu, expected_refiner_cu, diagnostics=True):
        _require(not self.closed and self._active_log.get() is None, 'Router closed or nested forward context')
        log = AttentionCallLog(self.mode, expected_cu, expected_refiner_cu, diagnostics)
        token = self._active_log.set(log)
        try:
            yield log
            log.finish()
        finally:
            self._active_log.reset(token)

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        log = self._active_log.get()
        _require(log is not None, 'Attention helper used outside the bound forward context')
        block = self._active_main.get()
        expected = log.expected_refiner_cu if block is None else log.expected_cu
        _require(q.shape == k.shape == v.shape == (expected[-1], HEADS, HEAD_DIM),
                 'Unexpected actual H3 QKV shape')
        _require(all(t.dtype == torch.bfloat16 and t.device == q.device for t in (q, k, v)),
                 'Original H3 QKV dtype/device changed')
        _require(self._cpu_fixture or q.device.type == 'cuda', 'Production attention requires CUDA')
        _require(softmax_scale == EXPECTED_SCALE, 'Original H3 softmax scale changed')
        log.remember_cu(cu_seqlens, expected, 'refiner' if block is None else 'main')
        if block is None:
            log.refiner_helpers += 1
            log.original_bf16_segments += 1
            out = self.original_helper(q, k, v, cu_seqlens, softmax_scale)
            log.finite(out)
            return out
        n, total = expected[1], expected[2]
        row = {'block': block, 'input_shape': list(q.shape), 'valid_length': n,
               'padding_length': total-n, 'softmax_scale': softmax_scale, 'mode': self.mode,
               'per_block_mean': None if self.mode == 'bf16' else self.mode == 'block_mean'}
        log.main_rows.append(row)
        if self.mode == 'bf16':
            log.original_bf16_segments += 2
            out = self.original_helper(q, k, v, log.main_cpu_cu, softmax_scale)
            log.finite(out)
            return out
        per_block_mean = self.mode == 'block_mean'
        segment = tuple(t[:n].transpose(0, 1).unsqueeze(0).contiguous() for t in (q, k, v))
        for tensor in segment:
            log.finite(tensor)
        packed = self.official.nvfp4_attention_sm120_quantize_qkv(*segment, per_block_mean=per_block_mean)
        padded = ((n+127)//128)*128
        groups = padded//128 if per_block_mean else 1
        _require(len(packed) == 7 and packed[-1].dtype == torch.float32
                 and tuple(packed[-1].shape) == (1, HEADS, groups, padded), 'Official compact correction ABI changed')
        log.finite(packed[-1])
        result = self.official.nvfp4_attention_sm120_fwd(
            *packed, sm_scale=softmax_scale, causal=False, per_block_mean=per_block_mean,
            out_dtype=torch.bfloat16, return_lse=False, unpadded_k_len=n)
        _require(result.shape == (1, HEADS, padded, HEAD_DIM) and result.dtype == torch.bfloat16
                 and result.device == q.device, 'Official output ABI changed')
        log.fp4_calls += 1
        row.update(official_input_shape=list(segment[0].shape), official_output_shape=list(result.shape),
                   correction_shape=list(packed[-1].shape), correction_bytes=packed[-1].numel()*packed[-1].element_size(),
                   unpadded_k_len=n, causal=False, return_lse=False)
        out = torch.empty_like(q)
        out[:n] = result[0, :, :n, :].transpose(0, 1)
        # Local boundary maps the exact original independent padding segment.
        # Original helper accepts CPU cu because it only calls .tolist().
        out[n:] = self.original_helper(q[n:], k[n:], v[n:], log.padding_cpu_cu, softmax_scale)
        log.original_bf16_segments += 1
        log.finite(out)
        return out

    def close(self):
        if self.closed:
            return
        _require(self._active_log.get() is None, 'Cannot close an active forward context')
        _require(self.comfy._sdpa_varlen_attention == self._dispatch, 'Global helper changed while router installed')
        self.comfy._sdpa_varlen_attention = self.original_helper
        for module, original in self.original_forwards:
            module.forward = original
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def install_h3_fp4_attention(dit, mode='block_mean'):
    """Return installed router; close it or use it as a context manager."""
    return H3AttentionRouter(dit, mode)


def is_native_fp4_attention_kernel(name):
    """Profiler evidence, separate from native FP4 projection GEMMs."""
    return 'nvfp4_attention::attention_kernel_ws' in name and 'float_e2m1' in name


def cpu_routing_fixture():
    """Mock routing checks only: never executes official/GPU attention math."""
    import types
    from torch import nn
    _require(not torch.cuda.is_initialized(), 'CPU fixture initialized CUDA')
    torch.set_num_threads(6)
    evidence = []
    for mode in MODES:
        original_calls, official_calls = [], []
        def original(q, k, v, cu_seqlens, softmax_scale):
            bounds = cu_seqlens.tolist()
            original_calls.append({'bounds': bounds, 'shape': list(q.shape), 'scale': softmax_scale,
                                   'cu_identity': id(cu_seqlens), 'cu_device': str(cu_seqlens.device)})
            return q+k+v
        comfy = types.SimpleNamespace(_sdpa_varlen_attention=original)
        class FakeAttention(nn.Module):
            num_heads, head_dim, softmax_scale = HEADS, HEAD_DIM, EXPECTED_SCALE
            def forward(self, x, cu_seqlens):
                return comfy._sdpa_varlen_attention(x, 2*x, 3*x, cu_seqlens, self.softmax_scale)
        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                def block():
                    value = nn.Module()
                    value.attn = FakeAttention()
                    return value
                self.blocks = nn.ModuleList([block() for _ in range(50)])
                self.token_refiner = nn.Module()
                self.token_refiner.blocks = nn.ModuleList([block() for _ in range(2)])
            def forward(self, x, text, cu, refcu):
                for block in self.token_refiner.blocks:
                    block.attn(text, refcu)
                for block in self.blocks:
                    output = block.attn(x, cu)
                return output
        model = FakeDiT()
        current = {}
        def quantize(q, k, v, *, per_block_mean):
            _require(q.shape == k.shape == v.shape == (1, HEADS, 129, HEAD_DIM), 'Fixture valid slice changed')
            _require(all(t.is_contiguous() for t in (q, k, v)), 'Fixture official inputs are not contiguous')
            current['out'] = q+k+v+1
            groups = 2 if per_block_mean else 1
            correction = torch.zeros((1, HEADS, groups, 256), dtype=torch.float32)
            return (None, None, None, None, None, None, correction)
        def official_forward(*packed, **kwargs):
            _require(kwargs == {'sm_scale': EXPECTED_SCALE, 'causal': False, 'per_block_mean': mode == 'block_mean',
                                 'out_dtype': torch.bfloat16, 'return_lse': False, 'unpadded_k_len': 129},
                     'Fixture official flags/scale/N changed')
            official_calls.append(kwargs.copy())
            out = torch.full((1, HEADS, 256, HEAD_DIM), 999., dtype=torch.bfloat16)
            out[:, :, :129] = current['out']
            return out
        official = types.SimpleNamespace(nvfp4_attention_sm120_quantize_qkv=quantize,
                                         nvfp4_attention_sm120_fwd=official_forward)
        # Axis-dependent BF16 values expose token/head/channel permutations.
        def axis_values(tokens):
            token = torch.arange(tokens).reshape(-1, 1, 1)
            head = torch.arange(HEADS).reshape(1, -1, 1)
            channel = torch.arange(HEAD_DIM).reshape(1, 1, -1)
            return ((13*token+7*head+3*channel) % 64).float().div(256).bfloat16()
        x, text = axis_values(133), axis_values(7)
        cu, refcu = torch.tensor([0, 129, 133], dtype=torch.int32), torch.tensor([0, 7, 7], dtype=torch.int32)
        original_forwards = [b.attn.forward for b in model.blocks]
        with H3AttentionRouter(model, mode, _comfy=comfy, _official=official, _cpu_fixture=True) as router:
            with router.forward_context(expected_cu=cu.tolist(), expected_refiner_cu=refcu.tolist(), diagnostics=True) as log:
                output = model(x, text, cu, refcu)
            expected = x+2*x+3*x
            if mode != 'bf16':
                expected[:129] += 1
            _require(torch.equal(output, expected), 'Fixture crop/tail output mapping failed')
            _require(len(official_calls) == (0 if mode == 'bf16' else 50), 'Fixture official call count changed')
            _require(sum(sum(b > a for a, b in zip(c['bounds'], c['bounds'][1:])) for c in original_calls)
                     == log.summary['original_bf16_segments'], 'Fixture actual BF16 segment count differs')
            bound_cpu = log.main_cpu_cu if mode == 'bf16' else log.padding_cpu_cu
            _require(all(c['cu_identity'] == id(bound_cpu) and c['cu_device'] == 'cpu' for c in original_calls[2:]),
                     'Fixture did not use the shared prebound CPU metadata policy')
            if mode != 'bf16':
                _require(all(c['bounds'] == [0, 4] and c['shape'][0] == 4 for c in original_calls[2:]),
                         'Fixture tail is not isolated original helper')
            mode_evidence = {k: v for k, v in log.summary.items() if k != 'main_rows'}
            with router.forward_context(expected_cu=cu.tolist(), expected_refiner_cu=refcu.tolist(), diagnostics=False) as timed_log:
                timed_output = model(x, text, cu, refcu)
            _require(torch.equal(timed_output, output) and timed_log.summary['finite_flag_count'] == 0,
                     'Diagnostic-free routing changed outputs or collected finite flags')
            mode_evidence['diagnostics_disabled_output_exact'] = True
            mode_evidence['common_cpu_metadata_identity_checked'] = True
            evidence.append(mode_evidence)
            # Wrong bound length must fail even without diagnostic finite ops.
            try:
                with router.forward_context(expected_cu=(0, 128, 133), expected_refiner_cu=refcu.tolist(), diagnostics=False):
                    model(x, text, cu, refcu)
            except RuntimeError:
                pass
            else:
                raise RuntimeError('Fixture wrong boundary was accepted')
        _require(comfy._sdpa_varlen_attention is original and
                 all(b.attn.forward == f for b, f in zip(model.blocks, original_forwards)), 'Router restoration failed')
    _require(is_native_fp4_attention_kernel('nvfp4_attention::attention_kernel_ws<cutlass::float_e2m1_t>')
             and not is_native_fp4_attention_kernel('cutlass3x_sm120_bstensorop_e2m1_gemm'), 'Kernel evidence classification failed')
    _require(not torch.cuda.is_initialized(), 'CPU fixture initialized CUDA')
    return {'status': 'complete', 'cuda_initialized': False, 'modes': evidence,
            'checks': ['original bound forwards restored', 'valid/padding/refiner routes', 'exact scale and N flags',
                       'axis-dependent BF16 token/head/channel values', 'contiguous official QKV',
                       'padded Q output cropped', 'wrong boundary rejected', 'separate native attention kernel matcher'],
            'scope': 'Mock routing only; no actual official quantizer, GPU kernel, or numerical accuracy proof'}


if __name__ == '__main__':
    import argparse
    import traceback
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-fixture-output', type=Path, required=True)
    args = parser.parse_args()
    _require(not args.cpu_fixture_output.exists(), 'Refusing existing fixture report')
    report = {'status': 'running', 'source': _source(__file__)}
    try:
        report.update(cpu_routing_fixture())
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        args.cpu_fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.cpu_fixture_output.write_text(json.dumps(report, indent=2, default=str)+'\n')
