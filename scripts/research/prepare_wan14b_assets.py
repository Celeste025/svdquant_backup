#!/usr/bin/env python3
"""Bounded, tensor-free download of pinned official Wan14B transformer assets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / 'results/research/asset_wan14b'
DATA = Path('/data1/models/svdquant-wjq')
REV = '38ec498cb3208fb688890f8cc7e94ede2cbd7f68'
REPO = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
TARGET = DATA / ('models/Wan2.1-T2V-14B-Diffusers-' + REV[:8])
WORK = DATA / 'research/cache/wan14b-assets-38ec498c'
LOG = ROOT / 'results/logs/wan14b_assets_download.log'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def hashes(path):
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            sha.update(b)
            git.update(b)
    return sha.hexdigest(), git.hexdigest()


def selected(info):
    metadata = {'.gitattributes', 'README.md', 'model_index.json',
                'scheduler/scheduler_config.json', 'text_encoder/config.json', 'vae/config.json'}
    return [x for x in info['siblings'] if x['rfilename'].startswith('transformer/') or x['rfilename'] in metadata]


def verify_headers(report):
    index = json.loads((TARGET / 'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
    names, dtype_counts, payload_bytes, elements = {}, {}, 0, 0
    for row in report['files'].values():
        path = Path(row['file'])
        if path.suffix != '.safetensors':
            continue
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            assert 0 < n < 16 * 1024**2
            head = json.loads(f.read(n))
        save(RD / 'headers' / (path.name + '.json'), head)
        entries = {k: v for k, v in head.items() if k != '__metadata__'}
        offset = 0
        for name, item in sorted(entries.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            begin, end = item['data_offsets']
            assert begin == offset and end >= begin, (path.name, name, begin, offset)
            width = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}[item['dtype']]
            count = math.prod(item['shape'])
            assert end - begin == count * width
            assert name not in names and index['weight_map'][name] == path.name
            names[name] = path.name
            dtype_counts[item['dtype']] = dtype_counts.get(item['dtype'], 0) + count
            elements += count
            offset = end
        assert 8 + n + offset == path.stat().st_size
        payload_bytes += offset
        row['safetensors'] = {'header_bytes': n, 'tensors': len(entries), 'payload_bytes': offset}
    assert names == index['weight_map']
    assert payload_bytes == index['metadata']['total_size']
    report['header_validation'] = {'status': 'complete', 'tensors': len(names), 'elements': elements,
                                  'elements_by_dtype': dtype_counts, 'payload_bytes': payload_bytes,
                                  'index_exact': True, 'tensor_loaded': False}


def worker(deadline):
    info = json.loads((RD / 'hub_model_info_pinned.json').read_text())
    assert info['sha'] == REV
    inventory = selected(info)
    result = RD / 'download.json'
    assert not result.exists() and not TARGET.exists(), 'Existing asset attempt must be preserved'
    assert shutil.disk_usage(DATA).free > 2 * sum(x['size'] for x in inventory)
    TARGET.mkdir(parents=True)
    WORK.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', repo=REPO, revision=REV, worker_pid=os.getpid(),
                  target=str(TARGET), cache=str(WORK), start_epoch=time.time(), deadline_epoch=deadline,
                  scope='Transformer plus small metadata only; no text-encoder/VAE weights, no complete-pipeline validation',
                  gpu_calls=0, tensor_loads=0, files={}, active={}, failures=[],
                  expected_download_bytes=sum(x['size'] for x in inventory),
                  model_card_license=info.get('cardData', {}).get('license'),
                  standalone_license_in_hf_tree=any('license' in x['rfilename'].lower() for x in info['siblings']))
    active = []

    def checkpoint():
        report['seconds'] = time.time() - report['start_epoch']
        for name, row in report['active'].items():
            p = Path(row['partial'])
            row['partial_bytes'] = p.stat().st_size if p.exists() else 0
        save(result, report)

    def finish(item, partial):
        size = partial.stat().st_size
        assert size == item['size'], (item['rfilename'], size, item['size'])
        sha, git = hashes(partial)
        if item.get('lfs'):
            assert sha == item['lfs']['sha256'], item['rfilename']
        else:
            assert git == item['blobId'], item['rfilename']
        dest = TARGET / item['rfilename']
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        partial.replace(dest)
        report['files'][item['rfilename']] = {'file': str(dest), 'bytes': size, 'sha256': sha,
                                            'git_blob_sha1': git, 'hub_entry': item, 'status': 'verified'}
        report['active'].pop(item['rfilename'], None)
        print(json.dumps({'event': 'verified', 'path': item['rfilename'], 'bytes': size}), flush=True)
        checkpoint()

    try:
        queue = list(inventory)
        while queue or active:
            if time.time() >= deadline:
                raise TimeoutError('Download deadline')
            while queue and len(active) < 2:
                item = queue.pop(0)
                rel = item['rfilename']
                partial = WORK / (rel.replace('/', '__') + '.partial')
                cached = RD / 'metadata' / rel
                if cached.exists():
                    assert not partial.exists()
                    shutil.copyfile(cached, partial)
                    finish(item, partial)
                    continue
                log = WORK / (rel.replace('/', '__') + '.curl.log')
                stream = log.open('x')
                url = f'https://huggingface.co/{REPO}/resolve/{REV}/{rel}?download=true'
                cmd = ['curl', '--fail', '--location', '--silent', '--show-error', '--continue-at', '-',
                       '--retry', '3', '--retry-all-errors', '--retry-delay', '3', '--connect-timeout', '30',
                       '--max-time', str(max(1, int(deadline-time.time()))), '--speed-limit', '1024',
                       '--speed-time', '90', '--output', str(partial), url]
                p = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
                active.append((p, stream, item, partial))
                report['active'][rel] = {'pid': p.pid, 'partial': str(partial), 'log': str(log), 'url': url}
                print(json.dumps({'event': 'download_started', 'path': rel, 'pid': p.pid}), flush=True)
            for job in list(active):
                p, stream, item, partial = job
                if p.poll() is not None:
                    stream.close()
                    active.remove(job)
                    assert p.returncode == 0, (item['rfilename'], p.returncode)
                    finish(item, partial)
            checkpoint()
            if active:
                time.sleep(3)
        verify_headers(report)
        report['status'] = 'complete'
        report['downloaded_bytes'] = sum(x['bytes'] for x in report['files'].values())
        checkpoint()
    except BaseException:
        report['status'] = 'failed_partial_preserved'
        report['error'] = traceback.format_exc()
        checkpoint()
        raise
    finally:
        for p, stream, _, _ in active:
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            stream.close()


def supervise(seconds):
    assert 0 < seconds <= 10800
    result = RD / 'supervisor.json'
    assert not result.exists() and not LOG.exists()
    WORK.mkdir(parents=True, exist_ok=True)
    for name in ['tmp', 'hf', 'xdg']:
        (WORK / name).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TMPDIR=str(WORK / 'tmp'), HF_HOME=str(WORK / 'hf'),
               XDG_CACHE_HOME=str(WORK / 'xdg'), HF_HUB_CACHE=str(WORK / 'hf/hub'),
               HF_XET_CACHE=str(WORK / 'hf/xet'), OMP_NUM_THREADS='2')
    start = time.time()
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--deadline-unix', str(start + seconds)]
    state = dict(status='running', command=command, supervisor_pid=os.getpid(), start_epoch=start,
                 deadline_epoch=start+seconds, log=str(LOG), cache=str(WORK), cuda_visible_devices='')
    with LOG.open('x') as f:
        p = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        state['worker_pid'] = p.pid
        save(result, state)
        try:
            rc = p.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(p.pid, signal.SIGTERM)
            try:
                rc = p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); rc = p.wait()
            state['deadline_or_interrupt'] = True
    state.update(returncode=rc, status='complete' if rc == 0 else 'failed_preserved', seconds=time.time()-start)
    save(result, state)
    raise SystemExit(rc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--seconds', type=int, default=10800)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if args.supervise:
        supervise(args.seconds)
    else:
        assert args.deadline_unix is not None
        worker(args.deadline_unix)
