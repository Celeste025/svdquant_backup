#!/usr/bin/env python3
"""Validate the resumable 16-prompt/20-timestep raw H3 calibration cache."""
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--cache-dir', type=Path, required=True)
a = p.parse_args(); cfg = json.loads(a.config.read_text())
expected_ids, expected_steps = list(map(int, cfg['prompt_ids'])), list(map(int, cfg['capture_steps']))
seen = set()
for pid in expected_ids:
    manifest_path = a.cache_dir / f'p{pid}' / 'manifest.json'
    if not manifest_path.is_file(): raise RuntimeError(f'missing manifest: {manifest_path}')
    m = json.loads(manifest_path.read_text())
    rows = m.get('samples', [])
    steps = sorted(int(r['step']) for r in rows)
    if steps != expected_steps: raise RuntimeError(f'p{pid}: expected steps {expected_steps}, got {steps}')
    for r in rows:
        key = (pid, int(r['step']))
        if key in seen: raise RuntimeError(f'duplicate {key}')
        seen.add(key)
        f = manifest_path.parent / r['file']
        if not f.is_file() or f.stat().st_size == 0: raise RuntimeError(f'missing/empty {f}')
if len(seen) != len(expected_ids) * len(expected_steps): raise RuntimeError('wrong call count')
print(json.dumps({'valid': True, 'prompts': len(expected_ids), 'calls': len(seen)}))
