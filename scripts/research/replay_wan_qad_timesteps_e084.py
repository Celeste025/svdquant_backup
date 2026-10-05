#!/usr/bin/env python3
"""Replay the immutable E022 test set and separate same-state from rollout error."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'third_party/deepcompressor'))
from generate_wan_qad_comparison import RCM, sha256, save_json

RD = ROOT / 'results/research/E084'
OLD = ROOT / 'results/research/E022'
DATA = Path('/data1/models/svdquant-wjq/research/20261005/E084')


def main():
    import torch
    import torch.nn.functional as F
    import diffusers
    from diffusers import WanTransformer3DModel
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import wan_mainweight_qad as qad
    import wan_native_nvfp4 as native
    from wan_nvfp4_fastpack import collect_fastpack_checks

    assert torch.__version__ == '2.11.0+cu128' and diffusers.__version__ == '0.33.1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert torch.cuda.get_device_capability() == (12, 0)
    assert not (RD / 'run.json').exists() and not DATA.exists()
    DATA.mkdir(parents=True)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    report = dict(experiment='E084', status='running', started=start, data=str(DATA),
                  environment=dict(python=sys.executable, torch=torch.__version__,
                                   diffusers=diffusers.__version__, gpu=torch.cuda.get_device_name()),
                  counts=dict(dit=0, native_mm=0, sdpa=0), arms={}, sources={})
    counts = report['counts']
    baseline = json.loads((OLD / 'video_baselines.json').read_text())
    selected = json.loads((OLD / 'video_selected.json').read_text())
    binding = json.loads((OLD / 'video_checkpoint_binding.json').read_text())
    assert baseline['status'] == selected['status'] == binding['status'] == 'complete'
    assert binding['selected_step'] == 64
    checkpoints = {'plain': baseline['plain_step0000_binding']['packed_artifact'],
                   'qad': binding['selected_packed_artifact']}
    source_files = [Path(__file__), Path(qad.__file__), Path(native.__file__),
                    ROOT / 'scripts/research/wan_nvfp4_fastpack.py',
                    ROOT / 'research_state/06_experiments/E084_wan_qad_timestep_replay_plan.md',
                    OLD / 'video_baselines.json', OLD / 'video_selected.json', OLD / 'video_checkpoint_binding.json']
    for p in source_files:
        report['sources'][str(p)] = sha256(p)
    for p in [Path(qad.__file__), Path(native.__file__), ROOT / 'scripts/research/wan_nvfp4_fastpack.py']:
        assert report['sources'][str(p)] == baseline['sources'][str(p)], f'Historical source changed: {p}'
    for artifact in checkpoints.values():
        assert sha256(artifact['path']) == artifact['sha256']
    report['checkpoints'] = checkpoints
    old_arms = {'bf16': baseline['arms']['bf16']['trajectories'],
                'plain': baseline['arms']['plain_step0000']['trajectories'],
                'qad': selected['arms']['qad_native_dev_selected']['trajectories']}
    tids = list(old_arms['bf16'])
    assert len(tids) == 16 and all(set(a) == set(tids) for a in old_arms.values())
    ts = torch.tensor(baseline['schedule']['t_steps'], dtype=torch.float64, device='cuda')
    ones = torch.ones(1, dtype=torch.float64, device='cuda')
    report['schedule'] = baseline['schedule']

    def checkpoint():
        report['seconds'] = time.time() - start
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
        save_json(RD / 'run.json', report)
        assert report['seconds'] < 2700, '45-minute guard'
        assert report['peak_allocated_gib'] < 60, 'Memory guard'

    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

    def metric(x, ref):
        x, ref = x.double(), ref.double()
        sse = (x - ref).square().sum().item()
        ref2 = ref.square().sum().item()
        return dict(sse=sse, ref2=ref2, nmse=sse / ref2)

    def interaction(a, b):
        a, b = a.double(), b.double()
        a2, b2, dot = a.square().sum().item(), b.square().sum().item(), (a*b).sum().item()
        return dict(a2=a2, b2=b2, twice_dot=2*dot,
                    cosine=dot/(a2*b2)**0.5 if a2*b2 else None,
                    net_interaction_fraction=2*dot/(a2+b2) if a2+b2 else None)

    original_sdpa, original_mm = F.scaled_dot_product_attention, F.scaled_mm
    def sdpa(q, k, v, *args, **kwargs):
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        counts['sdpa'] += 1
        return original_sdpa(q, k, v, *args, **kwargs)
    def mm(a, b, *args, **kwargs):
        assert a.dtype == b.dtype == torch.float4_e2m1fn_x2
        counts['native_mm'] += 1
        return original_mm(a, b, *args, **kwargs)
    F.scaled_dot_product_attention, F.scaled_mm = sdpa, mm
    try:
        with torch.inference_mode():
            for name in ('bf16', 'plain', 'qad'):
                model = WanTransformer3DModel.from_pretrained(RCM, torch_dtype=torch.bfloat16,
                                                             local_files_only=True).to('cuda').eval()
                installed = {}
                if name != 'bf16':
                    packed = torch.load(checkpoints[name]['path'], map_location='cpu', weights_only=False)
                    installed = qad.install_packed(model, packed)
                    del packed
                expected = 0 if name == 'bf16' else 300
                assert len([m for m in model.modules() if isinstance(m, native.NativeWanLinear)]) == expected
                assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
                report['arms'][name] = {}

                def forward(x, timestep, embedding):
                    before = dict(counts)
                    with collect_fastpack_checks() if expected else nullcontext([]) as checks:
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            v = model(hidden_states=x.to(torch.bfloat16), timestep=timestep,
                                      encoder_hidden_states=embedding, return_dict=False)[0]
                    assert v.dtype == torch.bfloat16 and torch.isfinite(v).all()
                    assert counts['sdpa'] - before['sdpa'] == 60
                    assert counts['native_mm'] - before['native_mm'] == expected and len(checks) == expected
                    counts['dit'] += 1
                    return v

                for tid in tids:
                    checkpoint()
                    old = old_arms[name][tid]
                    ref_old = old_arms['bf16'][tid]
                    shared_record = baseline['shared_inputs'][tid]['artifact']
                    assert sha256(shared_record['path']) == shared_record['sha256']
                    for key in ('initial_latent_sha256', 'noise_sha256', 'embedding_sha256'):
                        assert old[key] == ref_old[key]
                    payload = torch.load(shared_record['path'], map_location='cpu', weights_only=False)
                    x = payload['initial_latent_fp64'].to('cuda').clone()
                    noises = [n.to('cuda') for n in payload['update_noises_fp32']]
                    embedding = payload['embedding_bf16'].to('cuda')
                    assert digest(x) == old['initial_latent_sha256']
                    assert digest(embedding) == old['embedding_sha256']
                    assert [digest(n) for n in noises] == old['noise_sha256']
                    assert torch.equal(payload['t_steps_fp64'], ts.cpu())
                    teacher = None if name == 'bf16' else torch.load(DATA / 'bf16' / (tid + '.pt'), map_location='cpu', weights_only=True)
                    trace, rows, previous_error = [], [], None
                    for step, (t, tn) in enumerate(zip(ts[:-1], ts[1:])):
                        timestep = (t.float()*ones*1000).to(torch.bfloat16)
                        assert timestep.item() == old['steps'][step]['timestep']
                        before_cpu = x.cpu()
                        v = forward(x, timestep, embedding)
                        v_cpu = v.cpu()
                        x = (1-tn)*(x-t*v.to(torch.float64))+tn*noises[step]
                        after_cpu = x.cpu()
                        assert torch.isfinite(x).all()
                        step_hash = digest(after_cpu)
                        assert step_hash == old['steps'][step]['latent_after_sha256'], f'Replay mismatch {name}/{tid}/{step}'
                        record = dict(before=before_cpu, velocity=v_cpu, after=after_cpu)
                        row = dict(step=step, timestep=timestep.item(), t=t.item(), t_next=tn.item(),
                                   replay_sha256=step_hash, replay_matches=True)
                        if teacher is not None:
                            b = teacher[step]
                            same_v = v_cpu if step == 0 else forward(b['before'].to('cuda'), timestep, embedding).cpu()
                            record['same_state_velocity'] = same_v
                            a = same_v.double()-b['velocity'].double()
                            drift = v_cpu.double()-same_v.double()
                            total = v_cpu.double()-b['velocity'].double()
                            dx = before_cpu-b['before']
                            row.update(same_state=metric(same_v, b['velocity']),
                                       rollout_velocity=metric(v_cpu, b['velocity']),
                                       latent_before=metric(before_cpu, b['before']),
                                       latent_after=metric(after_cpu, b['after']),
                                       prediction_decomposition=interaction(a, drift))
                            assert torch.equal(a+drift, total)
                            c, d = (1-tn.item())*dx, -(1-tn.item())*t.item()*total
                            actual = after_cpu-b['after']
                            closure = (actual-c-d).abs().max().item()
                            assert closure < 1e-10
                            row['update_decomposition'] = interaction(c, d)
                            row['update_closure_maxabs'] = closure
                            if previous_error is not None:
                                ac = a-a.mean(dim=(2,3,4), keepdim=True)
                                pc = previous_error-previous_error.mean(dim=(2,3,4), keepdim=True)
                                row['adjacent_same_state_error'] = dict(raw=interaction(previous_error, a),
                                                                        channel_centered=interaction(pc, ac))
                            previous_error = a
                        trace.append(record)
                        rows.append(row)
                    assert sha256(old['final_latent']['path']) == old['final_latent']['sha256']
                    old_final = torch.load(old['final_latent']['path'], map_location='cpu', weights_only=True)
                    assert torch.equal(after_cpu, old_final)
                    path = DATA / name / (tid+'.pt')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    assert not path.exists()
                    torch.save(trace, path)
                    report['arms'][name][tid] = dict(prompt_id=old['prompt_id'], replica=old['replica'],
                        seed=old['seed'], steps=rows, final_exact_match=True,
                        trace=dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size))
                    checkpoint()
                    print(json.dumps(dict(arm=name, trajectory=tid, dit=counts['dit'], seconds=time.time()-start)), flush=True)
                    del teacher, trace, payload, x, noises, embedding, v, previous_error, old_final
                del model, installed
                gc.collect()
                torch.cuda.empty_cache()
        assert counts == dict(dit=288, native_mm=67200, sdpa=17280), counts
        report['status'] = 'complete'
        checkpoint()
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        save_json(RD / 'run.json', report)
        raise
    finally:
        F.scaled_dot_product_attention, F.scaled_mm = original_sdpa, original_mm


if __name__ == '__main__':
    main()
