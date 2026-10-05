#!/usr/bin/env python3
"""E010 auxiliary video-only diagnostic using the local official VR functions.

No generation source is modified. --check is CPU-only and does not load weights.
Runtime retains the official first-token decoding and all 29 unmodified queries.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = Path('/data1/models/svdquant-wjq/third_party/VisionReward')
MODEL = Path('/data1/models/svdquant-wjq/models/VisionReward-Video')
AUDIT = ROOT/'research_state/06_experiments/results/visionreward_contract_audit.json'
MANIFEST = ROOT/'research_state/06_experiments/E010_h3_heldout_manifest.json'
MEDIA = Path('/data1/models/svdquant-wjq/research/20261002/E010/decode')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(16 << 20), b''):
            h.update(part)
    return h.hexdigest()


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def score_answers(answers, weights):
    """Unknown text invalidates either usable score, never silently counts as no."""
    import numpy as np
    normalized = [a.strip().casefold() for a in answers]
    invalid = [i for i, a in enumerate(normalized) if a not in ('yes', 'no')]
    raw_official = float(np.mean(np.asarray([1 if a == 'yes' else -1 for a in answers])*weights))
    return {
        'strict_official': {'valid': not invalid, 'score': None if invalid else raw_official,
            'raw_formula_value_even_if_invalid': raw_official,
            'formula': 'mean(weight * (1 if first_token_decoded == "yes" else -1))',
            'exact_yes_count': sum(a == 'yes' for a in answers),
            'invalid_question_indices_zero_based': invalid},
        'normalized_yes_no': {'valid': not invalid,
            'score': None if invalid else float(np.mean(np.asarray([1 if a == 'yes' else -1 for a in normalized])*weights)),
            'normalization': 'first_token_decoded.strip().casefold()',
            'yes_count': normalized.count('yes'), 'no_count': normalized.count('no'),
            'invalid_question_indices_zero_based': invalid}}


def official_functions(namespace):
    """Execute only the two inspected function definitions, never top-level I/O."""
    source = OFFICIAL/'inference-video.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('load_video', 'inference')]
    if {n.name for n in nodes} != {'load_video', 'inference'}:
        raise RuntimeError('Official function definitions changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def provenance(report, full_hash=False):
    import numpy as np
    official = OFFICIAL/'inference-video.py'
    questions_path = OFFICIAL/'VisionReward_Video/VisionReward_video_qa_select.txt'
    weights_path = OFFICIAL/'VisionReward_Video/weight.json'
    audit = json.loads(AUDIT.read_text())
    for row in audit['source_files']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError(f'Previously inspected source changed: {row["path"]}')
    # readlines preserves every official trailing newline. Do not strip queries.
    with questions_path.open() as stream:
        questions = stream.readlines()
    weights = np.asarray(json.loads(weights_path.read_text()), dtype=np.float64)
    if len(questions) != 29 or weights.shape != (29,) or not np.isfinite(weights).all():
        raise RuntimeError('Expected all 29 official questions and finite weights')
    files = [Path(__file__), AUDIT, MANIFEST]+[Path(v['path']) for v in audit['source_files']]
    report['sources'] = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
    report['questions'] = questions
    report['weights'] = weights.tolist()
    report['model_files'] = {}
    shards = []
    for p in sorted(MODEL.iterdir()):
        if not p.is_file():
            continue
        info = {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
        if p.suffix == '.safetensors':
            shards.append(p.name)
            with p.open('rb') as stream:
                n = struct.unpack('<Q', stream.read(8))[0]
                raw = stream.read(n)
            header = json.loads(raw)
            size = 8+n+max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
            if size != info['bytes']:
                raise RuntimeError(f'Truncated model shard: {p}')
            info['header_sha256'] = hashlib.sha256(raw).hexdigest()
        if p.suffix != '.safetensors' or full_hash:
            info['sha256'] = sha(p)
        report['model_files'][str(p)] = info
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())
    if len(shards) != 6 or sorted(set(index['weight_map'].values())) != shards:
        raise RuntimeError('Model shard index mismatch')
    report['model_full_hash_complete'] = full_hash
    report['official_function_source'] = str(official)
    return questions, weights


def cpu_check(report, weights):
    import numpy as np
    import torch
    import transformers
    import decord
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    report['versions'] = {'python': sys.version, 'torch': torch.__version__,
                          'transformers': transformers.__version__, 'decord': decord.__version__}
    report['tokenizer_probe'] = {s: {'ids': tokenizer.encode(s),
        'decoded_first': tokenizer.decode(tokenizer.encode(s)[0])} for s in ('yes', 'Yes', 'no', 'No', ' yes')}
    fixture = score_answers(['yes', 'Yes', ' no '], np.ones(3))
    assert fixture['strict_official']['score'] == -1/3
    assert fixture['normalized_yes_no']['score'] == 1/3
    invalid = score_answers(['yes', 'maybe'], np.ones(2))
    assert invalid['strict_official']['score'] is None and invalid['normalized_yes_no']['score'] is None
    assert invalid['normalized_yes_no']['invalid_question_indices_zero_based'] == [1]
    captured = []
    class FakeReader:
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 124
        def get_frame_timestamp(self, indices): return [(i/24, (i+1)/24) for i in indices]
        def get_batch(self, indices):
            captured.extend(map(int, indices))
            return torch.zeros(len(indices), 2, 2, 3, dtype=torch.uint8)
    ns = official_functions({'io': io, 'np': np, 'torch': torch, 'cpu': lambda x: x,
        'VideoReader': FakeReader, 'bridge': type('Bridge', (), {'set_bridge': staticmethod(lambda _: None)})})
    frames = ns['load_video'](b'CPU synthetic frame-index fixture', strategy='chat')
    assert captured == [0, 24, 48, 72, 96, 120] and tuple(frames.shape) == (3, 6, 2, 2)
    report['cpu_contract'] = {'score_fixture': fixture, 'unknown_fixture': invalid,
                              'official_frame_fixture': captured, 'question_count': len(weights)}
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU check initialized CUDA')
    report['cuda_initialized'] = False


def load_cases(paths):
    manifest = json.loads(MANIFEST.read_text())
    expected = {r['prompt_id']: r for r in manifest['cases']}
    cases, seen = [], set()
    for path in paths:
        row = json.loads(path.read_text())
        key = (row['prompt_id'], row['variant'])
        source = expected[row['prompt_id']]
        if (key in seen or row['variant'] not in ('bf16', 'native') or row['status'] != 'complete'
                or row['prompt'] != source['prompt'] or row['seed'] != source['seed']):
            raise RuntimeError(f'Unexpected, duplicate or incomplete case: {path}')
        video = Path(row['video'])
        if not video.is_absolute() or sha(video) != row['video_sha256']:
            raise RuntimeError(f'Video content/path changed: {path}')
        seen.add(key)
        cases.append({**row, 'case_json': str(path.resolve()), 'case_json_sha256': sha(path),
            'case_id': f'p{row["prompt_id"]:03d}_seed{row["seed"]}_{row["variant"]}_{row["video_sha256"][:12]}'})
    if seen != {(p, a) for p in (30, 36) for a in ('bf16', 'native')}:
        raise RuntimeError('Require the four fixed E010 videos, with both arms for both prompts')
    return sorted(cases, key=lambda r: (r['prompt_id'], r['variant']))


def evaluate(args, report, questions, weights):
    import numpy as np
    import torch
    from decord import VideoReader, cpu, bridge
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cases = load_cases(args.case_json)
    snapshots = args.output.with_suffix('.sources')
    snapshots.mkdir(exist_ok=False)
    for path, info in report['sources'].items():
        target = snapshots/(info['sha256'][:12]+'_'+Path(path).name)
        shutil.copyfile(path, target)
        info['snapshot'] = str(target)
    save(report, args.output)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=dtype,
        trust_remote_code=True, local_files_only=True).eval().to('cuda')
    loaded_sources = {}
    prefix = type(model).__module__.rsplit('.', 1)[0]+'.'
    for name, module in list(sys.modules.items()):
        if name.startswith(prefix) and getattr(module, '__file__', None):
            path = Path(module.__file__)
            local = MODEL/path.name
            if path.suffix == '.py':
                if not local.is_file() or sha(path) != report['model_files'][str(local)]['sha256']:
                    raise RuntimeError(f'Loaded remote-code cache differs from local model source: {path}')
                loaded_sources[name] = {'path': str(path), 'sha256': sha(path), 'local_source': str(local)}
    if not loaded_sources:
        raise RuntimeError('Could not bind loaded model code to the local inspected sources')
    report['cpu_preflight_cuda_initialized'] = report.pop('cuda_initialized')
    report['runtime'] = {'dtype': str(dtype), 'device': torch.cuda.get_device_name(),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'generation_config': model.generation_config.to_dict(),
        'attention_implementation': getattr(model.config, '_attn_implementation', None),
        'loaded_model_sources': loaded_sources, 'cuda_initialized': torch.cuda.is_initialized()}
    traces, frame_traces = [], []
    class TracedReader:
        def __init__(self, *pos, **kw): self.reader = VideoReader(*pos, **kw)
        def __len__(self): return len(self.reader)
        def get_frame_timestamp(self, indices): return self.reader.get_frame_timestamp(indices)
        def get_batch(self, indices):
            frame_traces.append({'indices': list(map(int, indices)), 'total_frames': len(self.reader),
                'fps': float(self.reader.get_avg_fps()),
                'timestamps': self.reader.get_frame_timestamp(indices).tolist()})
            return self.reader.get_batch(indices)
    class TracedModel:
        def build_conversation_input_ids(self, **kw): return model.build_conversation_input_ids(**kw)
        def generate(self, **kw):
            output = model.generate(**kw)
            start = kw['input_ids'].shape[1]
            ids = output[0, start:].detach().cpu().tolist()
            if not ids: raise RuntimeError('Official generator returned no new token')
            traces.append({'input_token_ids': kw['input_ids'][0].detach().cpu().tolist(),
                'generated_token_ids': ids, 'first_token_id': ids[0],
                'first_token_decoded': tokenizer.decode(ids[0]), 'full_generated_decoded': tokenizer.decode(ids),
                'generation_kwargs': {k: kw[k] for k in ('max_new_tokens', 'pad_token_id', 'top_k', 'do_sample', 'top_p', 'temperature')}})
            return output
    ns = official_functions({'io': io, 'np': np, 'torch': torch, 'cpu': cpu, 'bridge': bridge,
        'VideoReader': TracedReader, 'model': TracedModel(), 'tokenizer': tokenizer, 'TORCH_TYPE': dtype})
    report['results'] = []
    for case in cases:
        row = {**case, 'status': 'running', 'answers': []}
        report['results'].append(row)
        for index, question in enumerate(questions):
            query = question.replace('[[prompt]]', case['prompt'])
            before = len(traces), len(frame_traces)
            answer = ns['inference'](case['video'], query)
            if (len(traces), len(frame_traces)) != (before[0]+1, before[1]+1):
                raise RuntimeError('Unexpected official inference call structure')
            if answer != traces[-1]['first_token_decoded']:
                raise RuntimeError('Official answer differs from traced first token')
            normalized = answer.strip().casefold()
            row['answers'].append({'question_index': index, 'question': question, 'expanded_query': query,
                'weight': float(weights[index]), **traces[-1], 'frames': frame_traces[-1],
                'normalized': normalized, 'recognized_yes_no': normalized in ('yes', 'no')})
            save(report, args.output)
        row['scores'] = score_answers([a['first_token_decoded'] for a in row['answers']], weights)
        row['status'] = 'complete' if row['scores']['normalized_yes_no']['valid'] else 'invalid_answers'
        save(report, args.output)
        print(json.dumps({'case_id': row['case_id'], 'scores': row['scores']}, ensure_ascii=False), flush=True)
    report['status'] = 'complete' if all(r['status'] == 'complete' for r in report['results']) else 'complete_with_invalid_answers'
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--case-json', type=Path, action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT/'results/research'/('E010_visionreward_cpucheck.json' if args.check else 'E010_visionreward.json')
    args.case_json = args.case_json or [MEDIA/f'p{p:03d}_{a}.json' for p in (30, 36) for a in ('bf16', 'native')]
    if args.output.exists(): raise FileExistsError(f'Preserve existing report: {args.output}')
    report = {'experiment': 'E010', 'status': 'running', 'check_only': args.check,
        'scope': 'Two paired prompts, auxiliary video-only diagnostic; not quality equivalence or generalization; audio unassessed',
        'official_compatibility': 'Local official load_video/inference definitions executed unchanged; local model code is separately hashed',
        'unknown_policy': 'Both usable scores null if any unrecognized answer; raw literal official formula retained with explicit invalid flag',
        'expected_cases': [str(p) for p in args.case_json]}
    started = time.monotonic()
    try:
        questions, weights = provenance(report, full_hash=not args.check)
        cpu_check(report, weights)
        if args.check:
            report['case_files_present'] = {str(p): p.exists() for p in args.case_json}
            report['status'] = 'complete'
        else:
            evaluate(args, report, questions, weights)
    except BaseException as exc:
        report.update(status='failed_stop', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        save(report, args.output)
        print(json.dumps({'status': report['status'], 'report': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
