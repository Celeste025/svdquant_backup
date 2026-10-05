#!/usr/bin/env python3
"""Independent, six-thread CPU reduction of frozen E018 attention tensors.

Imports no model/probe runner. Packet bytes are checked, never decoded here.
All outputs are already inverse-permuted into original query coordinates.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import summarize_h3_plain_baseline as util

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E018'
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E018/independent_summary')
PLOT_PYTHON = '/data1/models/svdquant-wjq/conda-envs/svdquant-ptq/bin/python'
N, START, LENGTH, FRAME, FRAMES = 22539, 1227, 21312, 576, 37
BLOCKS, MODES, SHIFTS = (0, 24, 48), ('bf16', 'block_mean', 'global_mean'), (0, 64, 128)
MAIN = slice(START+2*FRAME, START+35*FRAME)
require, rec = util.require, util.tensor_record


def permutation(shift):
    p = torch.arange(N)
    p[START:] = torch.roll(p[START:], shift)
    return p


def alternating(values, frames):
    """Balanced parity contrast; a constant has amplitude zero even with 33 frames."""
    frames = torch.as_tensor(frames)
    require(values.ndim == frames.ndim == 1 and len(values) == len(frames), 'Parity shape')
    return float((values[frames % 2 == 0].mean()-values[frames % 2 == 1].mean())/2)


def boundary_windows():
    # All 64-token supports lie within the original-query primary mask.
    offsets = torch.arange(-32, 32)
    frames = torch.arange(3, 35)
    true = START+frames[:, None]*FRAME+offsets
    pseudo = true+256
    require(int(true.min()) >= MAIN.start and int(pseudo.max()) < MAIN.stop, 'Boundary support')
    require(torch.equal(true % 128, pseudo % 128), 'Pseudo boundary phase changed')
    require(not bool(torch.isin(true.reshape(-1), pseudo.reshape(-1)).any()), 'Window overlap')
    return frames, true, pseudo


def reduce_rows(reference, phase0, output):
    require(reference.shape == phase0.shape == output.shape and reference.ndim == 3, 'Output shape')
    require(reference.dtype == phase0.dtype == output.dtype == torch.bfloat16, 'Output dtype')
    keys = ('reference_energy', 'error_energy', 'phase0_error_energy', 'delta_energy',
            'cross2', 'error_energy_change', 'identity_residual', 'max_abs_error', 'max_abs_delta')
    rows = {k: torch.empty(len(reference), dtype=torch.float64) for k in keys}
    rows['changed_from_phase0'] = torch.empty(len(reference), dtype=torch.int64)
    rows['byte_changes_from_phase0'] = torch.empty(len(reference), dtype=torch.int64)
    for start in range(0, len(reference), 256):
        end = min(start+256, len(reference)); span = slice(start, end)
        r, z, o = (v[span].double() for v in (reference, phase0, output))
        require(all(bool(torch.isfinite(t).all()) for t in (r, z, o)), 'Nonfinite raw output')
        e0, e, delta = z-r, o-r, o-z
        sumrow = lambda t: t.flatten(1).sum(1)
        values = dict(reference_energy=sumrow(r.square()), error_energy=sumrow(e.square()),
                      phase0_error_energy=sumrow(e0.square()), delta_energy=sumrow(delta.square()),
                      cross2=2*sumrow(e0*delta), max_abs_error=e.abs().flatten(1).amax(1),
                      max_abs_delta=delta.abs().flatten(1).amax(1))
        values['error_energy_change'] = values['error_energy']-values['phase0_error_energy']
        values['identity_residual'] = values['error_energy_change']-values['delta_energy']-values['cross2']
        for key, value in values.items(): rows[key][span] = value
        rows['changed_from_phase0'][span] = (output[span] != phase0[span]).flatten(1).sum(1)
        rows['byte_changes_from_phase0'][span] = (output[span].contiguous().view(torch.uint8) !=
                                                phase0[span].contiguous().view(torch.uint8)).flatten(1).sum(1)
    magnitude = rows['error_energy']+rows['phase0_error_energy']+rows['delta_energy']+rows['cross2'].abs()
    require(bool((rows['identity_residual'].abs() <= 1e-10*magnitude+1e-20).all()), 'Energy identity failed')
    return rows


def scope_stats(rows, span):
    r = {k: v[span] for k, v in rows.items()}
    sums = {k: float(r[k].sum()) for k in ('reference_energy', 'error_energy', 'phase0_error_energy',
                                         'delta_energy', 'cross2', 'error_energy_change')}
    return dict(sums, nmse=util.ratio(sums['error_energy'], sums['reference_energy']),
                delta_nmse=util.ratio(sums['delta_energy'], sums['reference_energy']),
                error_nmse_change=util.ratio(sums['error_energy_change'], sums['reference_energy']),
                cross2_over_reference=util.ratio(sums['cross2'], sums['reference_energy']),
                max_abs_error=float(r['max_abs_error'].max()), max_abs_delta=float(r['max_abs_delta'].max()),
                changed_elements=int(r['changed_from_phase0'].sum()),
                changed_bytes=int(r['byte_changes_from_phase0'].sum()),
                identity_max_abs_residual=float(r['identity_residual'].abs().max()))


def describe_rows(rows):
    fields = ('reference_energy', 'error_energy', 'delta_energy', 'cross2', 'error_energy_change')
    video = {k: rows[k][START:].reshape(FRAMES, FRAME) for k in fields}
    frames = {k: v.sum(1) for k, v in video.items()}
    primary_frames = torch.arange(2, 35)
    boundary_frames, true, pseudo = boundary_windows()
    boundary = {}
    for key in ('error_energy', 'error_energy_change', 'delta_energy', 'cross2'):
        a, b = rows[key][true].mean(1), rows[key][pseudo].mean(1)
        boundary[key] = dict(true_per_query=a.tolist(), pseudo_per_query=b.tolist(),
                             paired_true_minus_pseudo=(a-b).tolist(), mean_true=float(a.mean()),
                             mean_pseudo=float(b.mean()), mean_paired_difference=float((a-b).mean()),
                             alternating_true=alternating(a, boundary_frames),
                             alternating_pseudo=alternating(b, boundary_frames))
    reference_per_frame = float(frames['reference_energy'][2:35].mean())
    return dict(scopes={name: scope_stats(rows, span) for name, span in
                       [('full_valid', slice(0, N)), ('text', slice(0, 813)), ('audio', slice(813, START)),
                        ('all_video', slice(START, N)), ('primary_frames_2_34', MAIN)]},
        per_frame={k: v.tolist() for k, v in frames.items()},
        per_frame_nmse=[util.ratio(float(e), float(r)) for e,r in zip(frames['error_energy'], frames['reference_energy'])],
        primary_alternating={key: dict(signed_energy_amplitude=alternating(frames[key][2:35], primary_frames),
            amplitude_over_mean_reference=util.ratio(alternating(frames[key][2:35], primary_frames), reference_per_frame))
            for key in ('error_energy', 'error_energy_change', 'delta_energy', 'cross2')},
        boundary=dict(frame_indices=boundary_frames.tolist(), offsets=[-32,31], pseudo_offset=256, fields=boundary),
        # 576 = 9*64: retain both physical raster position and original 128-group phase.
        mean_within_frame={k: v[2:35].mean(0).tolist() for k,v in video.items()},
        mean_original_group_phase={k: [float(rows[k][MAIN][torch.arange(MAIN.start, MAIN.stop) % 128 == i].mean())
                                      for i in range(128)] for k in fields})


def load_checked(files, artifact, signature=None):
    files.record(artifact)
    value = util.load_tensor_file(artifact['file'])
    if signature is not None: require(util.signature(value) == signature, f'Tensor SHA mismatch: {artifact["file"]}')
    return value


def provenance(files, report_dir):
    launcher, lr = files.json(report_dir/'launcher.json')
    frozen, fr = files.json(report_dir/'frozen_contract.json')
    require(launcher['freeze_sha256'] == fr['sha256'], 'Frozen contract binding')
    for path, digest in frozen['files'].items(): files.verify(path, digest)
    require(launcher['gpu'] == 5 and launcher['max_dit_calls'] == 1 and launcher['max_attention_probes'] == 27,
            'Resource/call protocol changed')
    require(launcher['seconds_total'] <= 900 and launcher['deadline_epoch'] == launcher['start_epoch']+900, 'Shared deadline')
    require([s['name'] for s in launcher['stages']] == ['capture','probe'], 'Unexpected stages')
    reports = {}
    for stage in launcher['stages']:
        require(stage['status'] == 'complete' and stage['returncode'] == 0 and stage['pid'] > 0, 'Stage failed')
        files.verify(stage['report'], stage['report_sha256'])
        data, _ = files.json(stage['report']); files.sources(data['sources'])
        require(data['deadline_unix'] == launcher['deadline_epoch'], 'Reset budget')
        reports[stage['name']] = data
    capture, probe = reports['capture'], reports['probe']
    files.record(probe['capture_reference'])
    require(probe['capture_reference']['sha256'] == launcher['stages'][0]['report_sha256'], 'Capture reference')
    require(capture['attempted_dit_calls'] == capture['complete_dit_calls'] == 1 and
            probe['attention_calls'] == 27 and probe['complete_dit_calls'] == 0, 'Call count mismatch')
    require(capture['raw_replay_exact'] and capture['velocity_replay_exact'], 'Source replay failed')
    require(capture['case'] == dict(prompt_id=36, seed=59526, step=14, arm='block_mean'), 'Wrong source state')
    runtime = capture['runtime_audit']
    require(all(runtime[k] == n for k,n in [('sdpa_calls',52),('scaled_mm_calls',200),('disk_loads',0)]), 'Capture runtime')
    require(capture['attention']['fp4_calls'] == 50 and capture['zero_sf_checks']['checked_calls'] == 200 and
            capture['zero_sf_checks']['invalid_calls'] == 0, 'Capture execution contract')
    replay = load_checked(files, capture['full_replay'])
    require(util.signature(replay['raw_outputs']) == capture['raw_outputs'] == capture['expected_raw_outputs'], 'Raw replay tensor SHA')
    require(util.signature(replay['velocities']) == capture['velocities'] == capture['expected_velocities'], 'Velocity replay tensor SHA')
    require(replay['actual_dit_inputs'] == capture['actual_dit_inputs'] == capture['expected_dit_inputs'], 'Actual source inputs')
    files.record(capture['source_report']); source = json.loads(Path(capture['source_report']['file']).read_text())
    case = next(c for c in source['cases'] if c['prompt_id'] == 36)
    call = next(c for c in case['dit_calls'] if c['step'] == 14)
    require(call['actual_dit_inputs'] == capture['actual_dit_inputs'] and call['raw_outputs'] == capture['raw_outputs'], 'E017 source call')
    for modality, row in capture['source_steps'].items():
        state = load_checked(files, row['file'], row['tensors'])
        require(rec(state['noise_pred']) == capture['velocities'][modality], 'Historical velocity binding')
        saved = next(s for s in case['steps'] if s['step'] == 14 and s['modality'] == modality)
        require(saved['sha256'] == row['file']['sha256'], 'Historical state selection')
    return capture, probe, dict(launcher=lr, frozen=fr, seconds=launcher['seconds_total'], complete_dit_calls=1, attention_calls=27)


def run(args, report):
    files = util.Files()
    capture, probe, evidence = provenance(files, args.report_dir)
    report.update(provenance=evidence, inherited_scope='E017 validated models/weights inherited; no full model weight rehash', blocks=[])
    require([c['block'] for c in capture['cases']] == [c['block'] for c in probe['cases']] == list(BLOCKS), 'Layer selection')
    args.data_dir.mkdir(parents=True, exist_ok=False)
    all_arrays, table = {}, []
    for source, case in zip(capture['cases'], probe['cases']):
        require(source['artifact'] == case['capture'], 'Capture artifact binding')
        payload = load_checked(files, case['capture'], source['tensors'])
        require(payload['valid_length'] == N and payload['video_start'] == START and payload['video_tokens'] == LENGTH and
                payload['scale'] == 128**-.5, 'Capture geometry')
        require(all(tuple(payload[k].shape) == (N,56,128) and payload[k].dtype == torch.bfloat16 for k in
                    ('q','k','v','router_output')), 'Capture tensor dimensions')
        require(torch.equal(payload['positions']['video'], torch.arange(START, N)), 'Actual video positions')
        load_checked(files, case['kv_packets'], case['kv_tensor_records'])
        require(case['official_global_packet_parity'] is True, 'Global official parity failed')
        require([(r['mode'],r['shift']) for r in case['outputs']] == [(m,s) for m in MODES for s in SHIFTS], 'Probe call set')
        outputs = {}
        for row in case['outputs']:
            key = row['mode'], row['shift']
            require(row['permutation'] == rec(permutation(row['shift'])), 'Query permutation changed')
            outputs[key] = load_checked(files, row['output']['artifact'], row['output']['tensor'])
            require(outputs[key].shape == payload['q'].shape, 'Output shape differs')
            if row['mode'] == 'bf16':
                require(row['actual_q_packets'] is None and not row['fixed_kv_storage'], 'BF16 packet metadata')
            else:
                require(row['fixed_kv_storage'] is True, 'K/V storage not fixed')
                packets = load_checked(files, row['q_packet_artifact'])
                require(util.signature(packets) == {k:row['actual_q_packets'][k] for k in ('q_fp4','q_scale')}, 'Actual Q packet SHA')
        require(rec(outputs['block_mean',0]) == rec(payload['router_output']), 'Captured output phase-zero replay')
        global_corrections = [r['actual_q_packets']['correction'] for r in case['outputs'] if r['mode']=='global_mean']
        require(global_corrections[0] == global_corrections[1] == global_corrections[2], 'Frozen global correction changed')
        block = dict(block=case['block'], phase0_router_byte_exact=True, official_global_packet_parity=True,
                     fixed_kv_storage_inherited=True, q_packet_bytes_verified=True, outputs=[])
        arrays = {}
        for mode in MODES:
            for shift in SHIFTS:
                rows = reduce_rows(outputs['bf16',0], outputs[mode,0], outputs[mode,shift])
                arrays[f'{mode}_{shift}'] = rows
                summary = describe_rows(rows)
                block['outputs'].append(dict(mode=mode,shift=shift,**summary))
                primary = summary['scopes']['primary_frames_2_34']
                table.append(dict(block=case['block'],mode=mode,shift=shift,**primary,
                    signed_alternating_error_change=summary['primary_alternating']['error_energy_change']['signed_energy_amplitude'],
                    alternating_error_change_over_mean_reference=summary['primary_alternating']['error_energy_change']['amplitude_over_mean_reference'],
                    true_minus_pseudo_mean_error_change=summary['boundary']['fields']['error_energy_change']['mean_paired_difference']))
        path = args.data_dir/f'block{case["block"]}_row_arrays.pt'; torch.save(arrays,path)
        block['row_arrays'] = files.verify(path); block['row_array_schema'] = util.signature(arrays)
        report['blocks'].append(block)
        all_arrays[case['block']] = arrays
        print(f'E018 independent block {case["block"]} complete',flush=True)
    table_path = args.report_dir/'independent_table.json'
    table_path.write_text(json.dumps(table,indent=2,allow_nan=False)+'\n'); report['table'] = files.verify(table_path)
    plot_payload = args.data_dir/'plot_payload.pt'
    torch.save(all_arrays,plot_payload); report['plot_payload'] = files.verify(plot_payload)
    figure = args.data_dir/'E018_query_phase.png'
    subprocess.run([args.plot_python,str(Path(__file__).resolve()),'--plot-only',str(plot_payload),'--figure',str(figure)],
                   check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    report['figure'] = files.verify(figure)
    report.update(status='complete', verified_files=len(files.checked), files=files.checked)


def plot(payload_path, figure_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arrays = util.load_tensor_file(payload_path)
    fig, axes = plt.subplots(3,3,figsize=(17,12),constrained_layout=True)
    bframes, true, pseudo = boundary_windows()
    for row, block in enumerate(BLOCKS):
        values = arrays[block]
        ax = axes[row,0]
        for mode, color in zip(MODES,('gray','tab:blue','tab:orange')):
            for shift, style in ((64,'-'),(128,'--')):
                y = values[f'{mode}_{shift}']['error_energy_change'][START:].reshape(FRAMES,FRAME).sum(1)
                ax.plot(range(2,35),y[2:35],style,color=color,label=f'{mode} {shift}',linewidth=1)
        ax.axhline(0,color='k',linewidth=.5); ax.set_title(f'Block {block}: paired frame error-energy change')
        ax.set_xlabel('Original latent frame'); ax.set_ylabel('E(shift) - E(0)')
        if row==0: ax.legend(fontsize=7,ncol=2)
        heat = values['block_mean_64']['error_energy_change'][START:].reshape(FRAMES,FRAME)[2:35].numpy()
        vmax = max(float(np.max(np.abs(heat))),1e-30)
        im = axes[row,1].imshow(heat,aspect='auto',origin='lower',extent=[0,576,1.5,34.5],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axes[row,1].set_title('Block mean 64−0, per-query energy change')
        axes[row,1].set_xlabel('Within-frame raster token'); axes[row,1].set_ylabel('Original latent frame')
        fig.colorbar(im,ax=axes[row,1],shrink=.8)
        ax = axes[row,2]
        d = values['block_mean_64']['error_energy_change']
        ax.plot(bframes,d[true].mean(1),label='True boundary ±32',marker='.',linewidth=1)
        ax.plot(bframes,d[pseudo].mean(1),label='Pseudo +256 ±32',marker='.',linewidth=1)
        ax.axhline(0,color='k',linewidth=.5); ax.set_title('Same group phase: true vs pseudo boundary')
        ax.set_xlabel('Original latent frame boundary'); ax.set_ylabel('Mean per-query E64−E0')
        if row==0: ax.legend(fontsize=8)
    fig.suptitle('E018: fixed QKV, query-phase intervention; one state, three correlated layers\nComplete operator effects; no temporal-perception or Q-only mechanism claim',fontsize=13)
    fig.savefig(figure_path,dpi=150); plt.close(fig)


def selftest():
    frames = torch.arange(2,35)
    require(alternating(torch.ones(33,dtype=torch.float64)*19,frames)==0, 'DC leakage')
    signal = torch.where(frames%2==0,5.,-5.).double()+19
    require(alternating(signal,frames)==5, 'Balanced parity amplitude')
    ref = torch.zeros((5,2,4),dtype=torch.bfloat16)
    zero = torch.ones_like(ref)*2
    out = torch.ones_like(ref)
    rows = reduce_rows(ref,zero,out)
    require(bool((rows['delta_energy']>0).all()) and bool((rows['error_energy_change']<0).all()), 'Delta is not damage')
    require(bool((rows['cross2']==-32).all()) and bool((rows['identity_residual']==0).all()), 'Cross-term cancellation')
    bframes,true,pseudo = boundary_windows()
    require(len(bframes)==32 and true.numel()==pseudo.numel()==2048, 'Boundary count')
    for shift in SHIFTS:
        p = permutation(shift); inv = torch.argsort(p)
        require(torch.equal(p[inv],torch.arange(N)), 'Inverse permutation')
        require(torch.equal(p[:START],torch.arange(START)), 'Other modalities moved')
        if shift==128:
            ids = torch.arange(MAIN.start,MAIN.stop); positions=inv[ids]
            require(torch.equal(p[positions//128*128],ids//128*128), '128 group membership')
    return dict(status='complete', tests=['33-frame constant/DC rejection','balanced parity amplitude',
        'nonzero delta with improved error and negative cross term','energy identity','same-phase boundary supports',
        'inverse identity/nonvideo fixed/128 interior group membership'], cuda_initialized=torch.cuda.is_initialized())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-dir',type=Path,default=RD)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--plot-only',type=Path)
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--plot-python',default=PLOT_PYTHON)
    args=parser.parse_args(); torch.set_num_threads(6)
    if args.plot_only:
        plot(args.plot_only,args.figure); require(not torch.cuda.is_initialized(),'Plot initialized CUDA'); return
    output=args.output or args.report_dir/('independent_selftest.json' if args.self_test else 'independent_summary.json')
    require(not output.exists(),f'Refusing overwrite: {output}')
    started=time.monotonic()
    report=dict(experiment='E018',status='running',source={str(Path(__file__).resolve()):util.file_sha(__file__),
        str(Path(util.__file__).resolve()):util.file_sha(util.__file__)},
        methods=dict(accumulation='FP64 CPU; sums over H,D per original query',threads=6,
            primary_frames=[2,34],boundary_frames=[3,34],boundary_support='[-32,31], pseudo +256; both wholly inside main mask',
            balanced_alternating='0.5*(mean_even - mean_odd), original latent frame indices',
            identity='E_shift-E_0 = ||O_shift-O_0||^2 + 2< O_0-O_BF16_0, O_shift-O_0 >',
            reference='BF16 shift0 for every mode and shift; output tensors already inverse-permuted',
            limits=['One selected state, three correlated layers; no statistical significance or quality inference',
                'Fixed K/V does not freeze P; complete operator intervention only',
                'Packet hashes checked; no Q decoding or individual Q/P attribution',
                'Frame boundary effects may include spatial raster reset, not perceptual temporal flicker']))
    try:
        if args.self_test: report.update(selftest())
        else: run(args,report)
        require(not torch.cuda.is_initialized(),'CPU reduction initialized CUDA')
    except BaseException as exc:
        report.update(status='failed_stop',error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        report.update(seconds_total=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized())
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('status','seconds_total','cuda_initialized')}),flush=True)

if __name__=='__main__': main()
