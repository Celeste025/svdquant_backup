#!/usr/bin/env python3
"""E069: frozen source/recipe semantics, not a numerical calibration experiment."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'results/research/E069'
UP = REPORTS / 'upstream'
DC = ROOT / 'third_party/deepcompressor'
PLAN = ROOT / 'research_state/06_experiments/E069_h3_recipe_contract_plan.md'
DEST = REPORTS / 'audit.json'
COMMIT = '69f3473f5e1c1504bae35cc50c7858ef900a9b17'


def require(value, message):
    if not value:
        raise AssertionError(message)


def record(path):
    path = Path(path).resolve()
    value = path.read_bytes()
    return dict(file=str(path), bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


def method(path, class_name, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return node


def pairs(path, values):
    node = copy.deepcopy(method(path, 'SmoothCalibConfig', 'get_alpha_beta_pairs'))
    require(not node.decorator_list, 'Unexpected method decorator')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    strategies = SimpleNamespace(Manual=object(), GridSearch=object())
    namespace = {'SearchBasedCalibStrategy': strategies}
    exec(compile(module, str(path), 'exec'), namespace)
    config = SimpleNamespace(alpha=values['alpha'], beta=values['beta'],
                             num_grids=values['num_grids'], strategy=strategies.GridSearch)
    result = namespace[node.name](config)
    require(len(set(result)) == len(result), 'Duplicate candidates')
    return result, dict(first_line=node.lineno, last_line=node.end_lineno,
                        extracted_method_sha256=hashlib.sha256(ast.unparse(node).encode()).hexdigest())


def legacy_pairs(path, grids):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_smooth_candidates')
    expr = next(n.value for n in node.body if isinstance(n, ast.Return))
    require(isinstance(expr, ast.ListComp), 'Legacy candidate expression changed')
    bases = {}
    for n in ast.walk(expr.elt):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'pow':
            if isinstance(n.func.value, ast.Name) and n.func.value.id in ('x_max', 'w_max'):
                bases[n.func.value.id] = copy.deepcopy(n.args[0])
    require(set(bases) == {'x_max', 'w_max'}, 'Cannot isolate exact legacy exponent expressions')
    selected = ast.Expression(body=ast.ListComp(elt=ast.Tuple(elts=[bases['x_max'], bases['w_max']], ctx=ast.Load()),
                                                generators=copy.deepcopy(expr.generators)))
    result = eval(compile(ast.fix_missing_locations(selected), str(path), 'eval'), {'grids': grids})
    return result, dict(first_line=node.lineno, last_line=node.end_lineno,
                        extracted_expression=ast.unparse(selected))


def main():
    require(not DEST.exists(), 'Preserve existing audit')
    started = time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    out = dict(experiment='E069', status='running', sources={}, new_model_forwards=0,
               new_calibration_runs=0, new_candidate_weights=0, torch_imported=False,
               script=record(__file__), plan=record(PLAN))
    try:
        source_manifest = json.loads((UP/'source_manifest.json').read_text())
        require(source_manifest['status'] == 'complete' and source_manifest['commit'] == COMMIT,
                'Pinned source retrieval incomplete')
        require(len(source_manifest['files']) <= 12 and source_manifest['total_bytes'] < 5*1024**2,
                'Source budget exceeded')
        out['upstream_manifest'] = record(UP/'source_manifest.json')
        for row in source_manifest['files']:
            actual = record(row['file'])
            require(actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'], 'Upstream source changed')
            out['sources'][actual['file']] = actual
            local = DC/row['relative_path']
            if local.is_file():
                out['sources'][str(local.resolve())] = record(local)
        rel = 'examples/diffusion/configs/svdquant/__default__.yaml'
        uq = yaml.safe_load((UP/rel).read_text())['quant']
        lq = yaml.safe_load((DC/rel).read_text())['quant']
        cfgrel = 'deepcompressor/calib/config/smooth.py'
        up_pairs, up_method = pairs(UP/cfgrel, uq['smooth']['proj'])
        local_pairs, local_method = pairs(DC/cfgrel, lq['smooth']['proj'])
        local10, _ = pairs(DC/cfgrel, dict(lq['smooth']['proj'], num_grids=10))
        legacy = ROOT/'scripts/ptq_minimax_h3_svdquant_standard.py'
        old_pairs, old_method = legacy_pairs(legacy, 10)
        out['sources'][str(legacy)] = record(legacy)
        require(up_pairs == local_pairs and len(up_pairs) == 39 and len(local10) == 19 and len(old_pairs) == 9,
                'Observed candidate contract changed')
        require(set(old_pairs) <= set(local10), 'Legacy exponent family no longer nested')
        out['smoothing'] = dict(upstream_default_config=uq['smooth']['proj'],
            upstream_pairs=up_pairs, upstream_method=up_method, vendored_default_pairs=local_pairs,
            vendored_method=local_method, vendored_grid10_pairs=local10,
            legacy_grid10_pairs=old_pairs, legacy_expression=old_method,
            missing_at_matched_grid10=[p for p in local10 if p not in old_pairs],
            counts=dict(upstream_default=39, vendored_grid10=19, legacy_grid10=9),
            interpretation='Legacy omits identity and activation-only exponent family, even at the same grid count.')
        out['low_rank_default'] = uq['wgts']['low_rank']
        out['decomposition_source'] = {}
        lrrel = 'deepcompressor/nn/patch/lowrank.py'
        for name, base in [('upstream', UP), ('vendored', DC)]:
            node = method(base/lrrel, 'LowRankBranch', 'reset_parameters')
            calls = [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Call) and
                     ast.unparse(n.func) in ('torch.linalg.svd', 'torch.svd_lowrank')]
            require(len(calls) == 1, 'Unexpected decomposition call set')
            out['decomposition_source'][name] = dict(call=calls[0], first_line=node.lineno, last_line=node.end_lineno)
        require(out['decomposition_source']['upstream']['call'] == 'torch.linalg.svd(weight.double())',
                'Pinned upstream SVD differs')
        require(out['decomposition_source']['vendored']['call'] == 'torch.svd_lowrank(weight.float(), q=q, niter=2)',
                'Vendored SVD differs')
        out['quantization_yaml'] = {
            'upstream_nvfp4': yaml.safe_load((UP/'examples/diffusion/configs/svdquant/nvfp4.yaml').read_text())['quant'],
            'local_real_nvfp4': yaml.safe_load((DC/'examples/diffusion/configs/svdquant/real_nvfp4.yaml').read_text())['quant']}
        extra = DC/'examples/diffusion/configs/svdquant/real_nvfp4.yaml'
        out['sources'][str(extra)] = record(extra)
        out['prior_result_bindings'] = [record(ROOT/p) for p in
            ['results/research/E060/audit.json', 'results/research/E065b/evaluate.json']]
        require('torch' not in sys.modules and time.monotonic()-started < 60, 'CPU-only audit contract violated')
        out.update(status='complete', seconds=time.monotonic()-started, upstream_commit=COMMIT,
            decision='E065b is a fixed-recipe diagnostic, not a complete upstream SVDQuant baseline.',
            limitations=['Pure candidate enumeration and source identity, no model performance tested.',
                         'Source differences do not establish better video quality for either recipe.',
                         'Current H3 source supports block scoring/student propagation; historical producer SHA is missing.',
                         'Upstream NVFP4 YAML and the local native two-level activation scale contract differ.',
                         'No new method, no fidelity claim for an unexecuted H3 adapter.'])
    except BaseException as exc:
        out.update(status='failed_preserved', error=repr(exc), seconds=time.monotonic()-started)
        raise
    finally:
        with DEST.open('x') as stream:
            json.dump(out, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k:out[k] for k in ('status','seconds','decision','new_model_forwards')}))


if __name__ == '__main__':
    main()
