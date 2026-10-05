#!/usr/bin/env python3
"""E060: read-only CPU audit of saved H3 factors and current LR source semantics.

No model/quantizer/calibrator is imported or executed. Export containers are
memory-mapped and only smooth/A/B data are inspected; packed bodies are not
hashed, decoded, or materialized by this script.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import torch

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'research_state/06_experiments/E060_h3_lowrank_provenance_plan.md'
OUTPUT = ROOT/'results/research/E060/audit.json'
EXPORT = Path('/data1/models/svdquant-wjq/research/20261002/E009/legacy_export')
MANIFEST = EXPORT/'manifest.json'
STATE = ROOT/'results/checkpoints/minimax_h3_svdquant_standard_8p64s/quant_state.pt'
EXPECTED_STATE_SHA = 'e5ccf2d809f0b03e6a0eaef400e4f610ccfc37478eb3715d75040cf07446652d'
STANDARD = ROOT/'scripts/ptq_minimax_h3_svdquant_standard.py'
NN = ROOT/'third_party/deepcompressor/deepcompressor/nn/patch/lowrank.py'
CALIBRATOR = ROOT/'third_party/deepcompressor/deepcompressor/calib/lowrank.py'
CALIB_CONFIG = ROOT/'third_party/deepcompressor/deepcompressor/calib/config/lowrank.py'
SUFFIXES = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')
TARGETS = {f'blocks.{i}.{suffix}' for i in range(50) for suffix in SUFFIXES}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    path = Path(path)
    st = path.stat()
    return dict(file=str(path), bytes=st.st_size, sha256=digest(path))


def excerpt(text, node):
    lines = text.splitlines()
    return dict(start_line=node.lineno, end_line=node.end_lineno,
                text='\n'.join(lines[node.lineno-1:node.end_lineno]))


def expr(node):
    return ast.unparse(node)


def source_audit(manifest):
    texts = {p: p.read_text() for p in (STANDARD, NN, CALIBRATOR, CALIB_CONFIG)}
    trees = {p: ast.parse(text, filename=str(p)) for p, text in texts.items()}
    candidate = next(n for n in trees[STANDARD].body
                     if isinstance(n, ast.FunctionDef) and n.name == '_candidate_lowrank')
    assignments = {expr(n.targets[0]): expr(n.value) for n in candidate.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1}
    assert assignments['base_q'] == 'nvfp4_qdq(ws)'
    assert assignments['qweight'] == 'nvfp4_qdq(ws - b @ a)'
    branch = next(n for n in ast.walk(candidate) if isinstance(n, ast.Call)
                  and expr(n.func) == 'LowRankBranch')
    branch_weight = expr(next(k.value for k in branch.keywords if k.arg == 'weight'))
    assert branch_weight == 'ws - base_q'
    seeds = [n for n in ast.walk(candidate) if isinstance(n, ast.Call)
             and expr(n.func) == 'torch.manual_seed']
    assert len(seeds) == 1 and expr(seeds[0].args[0]) == 'seed'
    parameters = [a.arg for a in candidate.args.args]
    assert parameters == ['ws', 'rank', 'seed']
    loop = next(n for n in ast.walk(trees[STANDARD]) if isinstance(n, ast.For)
                and expr(n.target) == 'candidate_id'
                and expr(n.iter) == 'range(1, args.max_lowrank_iters)')
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
             and expr(n.func) == '_candidate_lowrank']
    assert len(calls) == 1
    assert [expr(a) for a in calls[0].args[:2]] == ['ws', 'args.rank']
    assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                   and n.id == 'ws' for n in ast.walk(loop))
    assert any(isinstance(n, ast.Break) for n in ast.walk(loop))

    cls = next(n for n in trees[CALIBRATOR].body
               if isinstance(n, ast.ClassDef) and n.name == 'QuantLowRankCalibrator')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    reset, ask, tell = (methods[k] for k in ('_reset', '_ask', '_tell'))
    compensate = next(n for n in ast.walk(reset) if isinstance(n, ast.If)
                      and expr(n.test) == 'self.config.compensate')
    assert any(isinstance(n, ast.Call) and expr(n.func) == 'self.w_quantizer.quantize'
               for n in ast.walk(compensate))
    carry_branch = next(n for n in ast.walk(ask) if isinstance(n, ast.Call)
                       and expr(n.func) == 'LowRankBranch')
    carry_weight = expr(next(k.value for k in carry_branch.keywords if k.arg == 'weight'))
    assert carry_weight == 'self.w - self.qw'
    writes = [n for n in ast.walk(ask) if isinstance(n, ast.Assign)
              and any(expr(t) == 'self.qw' for t in n.targets)]
    assert len(writes) == 2
    assert any('quantize(self.w - lw' in expr(n.value) for n in writes)
    assert not any(isinstance(n, ast.Call) and expr(n.func).endswith('._reset')
                   for n in ast.walk(ask))

    branch_cls = next(n for n in trees[NN].body
                      if isinstance(n, ast.ClassDef) and n.name == 'LowRankBranch')
    reset_params = next(n for n in branch_cls.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'reset_parameters')
    svd = [n for n in ast.walk(reset_params) if isinstance(n, ast.Call)
           and expr(n.func) == 'torch.svd_lowrank']
    assert len(svd) == 1
    assert expr(next(k.value for k in svd[0].keywords if k.arg == 'niter')) == '2'
    q_assignment = next(n for n in ast.walk(reset_params) if isinstance(n, ast.Assign)
                        and any(expr(t) == 'q' for t in n.targets))

    # Preserve the export-time records separately from today's source hashes.
    current = {str(p): record(p) for p in texts}
    for path, historical in manifest['sources'].items():
        p = Path(path)
        if p.suffix == '.py' and p.is_file():
            current[str(p)] = record(p)
            current[str(p)]['export_recorded_sha256'] = historical['sha256']
            current[str(p)]['matches_export_recorded_sha256'] = (
                current[str(p)]['sha256'] == historical['sha256'])
    return dict(
        current_source_records=current,
        historical_ptq_source_binding={str(p): str(p) in manifest['sources']
                                      for p in (STANDARD, NN, CALIBRATOR)},
        standard_candidate=dict(parameters=parameters, base_quantizer_assignment=assignments['base_q'],
            branch_weight_argument=branch_weight, residual_quantizer_assignment=assignments['qweight'],
            rng_seed_argument=expr(seeds[0].args[0]), next_candidate_call=expr(calls[0]),
            ws_reassigned_in_candidate_loop=False,
            interpretation='Current helper restarts Q0=Q(Ws) on every call; loop changes the SVD seed, '
                           'without passing the preceding candidate quantized weight as state.',
            helper_source=excerpt(texts[STANDARD], candidate),
            loop_source=excerpt(texts[STANDARD], loop)),
        local_vendored_calibrator=dict(
            branch_weight_argument=carry_weight,
            next_quantized_weight_assignments=[excerpt(texts[CALIBRATOR], n) for n in writes],
            compensate_initialization_supported=True,
            interpretation='Current local vendored _reset supports quantization-residual initialization '
                           'when compensate=True; _ask subsequently reads and replaces persistent self.qw. '
                           'The initialization itself is not an error.',
            reset_source=excerpt(texts[CALIBRATOR], reset),
            ask_source=excerpt(texts[CALIBRATOR], ask),
            tell_source=excerpt(texts[CALIBRATOR], tell)),
        current_local_lowrank_svd=dict(
            call=expr(svd[0]), q_expression=expr(q_assignment.value),
            niter=2, reset_parameters_source=excerpt(texts[NN], reset_params),
            interpretation='Current local LowRankBranch uses randomized approximate torch.svd_lowrank. '
                           'This describes local code, not a verified upstream revision or a quality disadvantage.'),
        historical_production_semantics_proven=False,
        limitation='The export manifest binds export-time sources and the saved state, but does not bind '
                   'the standard PTQ producer, LowRankBranch, or calibrator source used to produce that state. '
                   'Current AST semantics and saved candidate-count patterns cannot prove historical execution.')


def factor_bytes(t):
    assert isinstance(t, torch.Tensor) and t.device.type == 'cpu'
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def audit(result):
    assert PLAN.is_file(), 'Root must fix the plan before execution'
    assert not torch.cuda.is_initialized()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'complete' and manifest['target_count'] == 200
    result['manifest'] = record(MANIFEST)
    result['plan'] = record(PLAN)
    result['script'] = record(Path(__file__))
    binding = manifest['sources'][str(STATE)]
    observed = record(STATE)
    assert observed['bytes'] == binding['bytes'] == 301607239
    assert observed['sha256'] == binding['sha256'] == EXPECTED_STATE_SHA
    result['state_binding'] = dict(observed=observed, export_manifest_record=binding,
                                   full_state_hash_verified=True)
    result['source_audit'] = source_audit(manifest)
    state = torch.load(STATE, map_location='cpu', weights_only=True, mmap=True)
    assert state['format'] == 'minimax-h3-svdquant-standard-v1'
    assert set(state['layers']) == TARGETS
    assert state['config'] == manifest['state_config']
    assert state['config']['rank'] == 32 and state['config']['group_size'] == 16
    result['saved_state_metadata'] = {k: state[k] for k in ('format', 'config', 'checks', 'completed_blocks')
                                      if k in state}
    exports = {r['name']: r for r in manifest['layers']}
    assert len(manifest['layers']) == len(exports) == 200 and set(exports) == TARGETS
    candidate_ids, used_counts, ranks = Counter(), Counter(), Counter()
    matched_fields = factor_read_bytes = full_export_bytes = 0
    result['layers'] = []
    for name in sorted(TARGETS):
        saved = state['layers'][name]
        rec = exports[name]
        path = EXPORT/rec['file']
        st = path.stat()
        assert st.st_size == rec['file_bytes'], (name, 'export size mismatch')
        full_export_bytes += st.st_size
        # mmap maps container storage without reading or decoding packed bodies.
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert payload['name'] == name and payload['format'] == manifest['format']
        assert payload['recipe'] == manifest['recipe']
        shape = list(saved['shape'])
        assert payload['shape'] == rec['shape'] == shape
        oc, ic = shape
        assert tuple(saved['smooth'].shape) == (ic,)
        assert tuple(saved['final_a'].shape) == (32, ic)
        assert tuple(saved['final_b'].shape) == (oc, 32)
        checks = {}
        for exported_key, state_key in (('smooth', 'smooth'), ('lr_a', 'final_a'), ('lr_b', 'final_b')):
            a, b = payload['tensors'][exported_key], saved[state_key]
            assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape
            ab, bb = factor_bytes(a), factor_bytes(b)
            exact = ab == bb
            if not exact:
                result['factor_mismatch'] = dict(layer=name, export_field=exported_key, state_field=state_key)
                raise AssertionError(result['factor_mismatch'])
            checks[exported_key] = dict(state_field=state_key, shape=list(a.shape), dtype=str(a.dtype),
                bytes=len(ab), byte_exact=True, exported_factor_sha256=hashlib.sha256(ab).hexdigest(),
                state_factor_sha256=hashlib.sha256(bb).hexdigest())
            factor_read_bytes += len(ab)
            matched_fields += 1
        cid, used = int(saved['candidate_id']), int(saved['candidates_used'])
        candidate_ids[cid] += 1
        used_counts[used] += 1
        ranks[int(saved['final_a'].shape[0])] += 1
        result['layers'].append(dict(name=name, shape=shape, rank=32, candidate_id=cid,
            candidates_used=used, candidates_used_equals_selected_plus_two=(used == cid+2),
            alpha=float(saved['alpha']), smooth_error=float(saved['smooth_error']),
            initial_error=float(saved['initial_error']), final_error=float(saved['final_error']),
            export_file=str(path), actual_export_file_bytes=st.st_size,
            manifest_expected_full_export_sha256=rec['file_sha256'],
            full_export_sha256_recomputed=False, factor_checks=checks))
        del payload, a, b, ab, bb
    result['metadata_counts'] = dict(layer_count=200,
        selected_candidate_id=dict(sorted(candidate_ids.items())),
        candidates_used=dict(sorted(used_counts.items())), rank=dict(sorted(ranks.items())),
        candidates_used_equals_selected_plus_two=sum(r['candidates_used_equals_selected_plus_two']
                                                     for r in result['layers']),
        configured_max_lowrank_iters=state['config']['max_lowrank_iters'],
        actual_max_candidates_used=max(used_counts),
        interpretation='Saved metadata are observations, not proof of the historical candidate-generation algorithm.')
    result['verification_scope'] = dict(
        matched_layers=200, matched_factor_fields=matched_fields,
        state_full_sha256_recomputed=True, exports_opened_with_mmap=200,
        exported_factor_bytes_compared=factor_read_bytes,
        total_export_container_bytes=full_export_bytes,
        all_export_full_hashes_recomputed=False,
        packed_weights_or_scales_materialized_or_decoded=False,
        original_model_checkpoint_loaded_or_hashed=False,
        hardware_io_bytes_measured=False,
        limitation='File-size checks and factor equality do not revalidate packed-weight/scales bytes. '
                   'The stored full-export hashes remain inherited records, not fresh verification. '
                   'mmap may cause OS readahead; only logical tensors inspected are claimed.')
    result['conclusion'] = (
        'All 200 native-export smooth/A/B factors match the SHA-bound legacy state byte for byte. '
        'Keep historical same-implementation results, but describe this recipe as a legacy '
        'smooth-plus-low-rank residual baseline. Current producer code restarts candidate residuals, '
        'whereas the local vendored calibrator carries quantized-weight state. Historical producer '
        'code identity is unproven. These findings do not establish inferior calibration quality '
        'or rule out what a properly matched standard SVDQuant calibration could accomplish.')
    result['limits'] = [
        'Exploratory baseline audit after prior read-only observations, not a blind test.',
        'No saved candidate sequence, original production source hash, or historical runtime was reconstructed.',
        'Compensation initialization is supported by the local vendored calibrator and is not itself a defect.',
        'No model forward, weight quantization, calibration, checkpoint mutation, or quality evaluation.',
        'No claim that random-restart candidates or approximate SVD necessarily worsen quality.',
    ]
    assert not torch.cuda.is_initialized()


def main():
    assert not OUTPUT.exists(), f'Refusing to overwrite {OUTPUT}'
    torch.set_num_threads(4)
    started = time.monotonic()
    result = dict(experiment='E060', status='running', cpu_budget_seconds=120,
                  new_model_forwards=0, checkpoint_writes=0)
    def timeout(_sig, _frame):
        raise TimeoutError('E060 120-second CPU wall budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        audit(result)
        result['status'] = 'complete'
    except Exception:
        result.update(status='failed_stop', error=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(seconds=time.monotonic()-started, cuda_initialized=torch.cuda.is_initialized())
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        metadata_counts=result.get('metadata_counts'), verification_scope=result.get('verification_scope'),
        output=str(OUTPUT)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
