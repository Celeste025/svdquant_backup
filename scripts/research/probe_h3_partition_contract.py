#!/usr/bin/env python3
"""E019 bounded fixed-packet partition contract; no DiT or new quantizer.

CPU decoding supplies a mathematical reference, not an emulation of native MMA,
online softmax, or internal P quantization. Only correction is regenerated on
GPU, using the original full-shape official preprocessing and an exact SHA gate.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import probe_h3_plain_baseline as base
import torch
from nvfp4_attention_fixed_scale import logical_scales, k_permutation

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E019'
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E019')
PRIOR = ROOT/'results/research/E018'
PLAN = ROOT/'research_state/06_experiments/E019_h3_partition_contract_plan.md'
BLOCKS = (0, 24, 48)
HEADS = (0, 8, 16, 24, 32, 40, 48, 55)
FRAMES, RASTER = (2, 18, 34), (0, 287, 575)
QUERIES = tuple(1227 + frame*576 + pixel for frame in FRAMES for pixel in RASTER)
N, NP, SPLIT, D = 22539, 22656, 11264, 128
SCALE = D**-.5
KV_KEYS = ('k_fp4', 'v_fp4_t', 'k_scale', 'v_scale_t')
Q_KEYS = ('q_fp4', 'q_scale')
NATIVE = ('bf16_full', 'fp16_full', 'fp16_left', 'fp16_right')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def complete(path):
    value = json.loads(Path(path).read_text())
    require(value.get('status') == 'complete', f'Incomplete prerequisite: {path}')
    return value


def equal_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def budget(args, report, before_call=False):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix, 'Missing/expired original shared deadline')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '5', 'Only physical GPU5 is authorized')
    if before_call:
        require(report['attention_calls'] < 12, '12-attention-call budget exhausted')
    require(torch.cuda.max_memory_allocated() <= 60*1024**3, '60-GiB peak allocation guard exceeded')


def slice_kv(packet, begin, end):
    """Copy physical tiles; no mean/scale calculation and no numeric encoding."""
    require(begin % 128 == end % 128 == 0 and 0 <= begin < end <= NP, 'Unaligned KV slice')
    length = end-begin
    sf = packet['v_scale_t']
    b, h, d, cols = sf.shape
    require(d == D and cols == NP//16, 'Wrong full V scale geometry')
    vsf = (sf.view(torch.uint8).reshape(b, h, d//64, NP//64, 256)
           [:, :, :, begin//64:end//64, :].contiguous()
           .reshape(b, h, d, length//16).view(torch.float8_e4m3fn))
    return dict(k_fp4=packet['k_fp4'][:, :, begin:end, :].contiguous(),
                k_scale=packet['k_scale'][:, :, begin:end, :].contiguous(),
                v_fp4_t=packet['v_fp4_t'][:, :, :, begin//2:end//2].contiguous(),
                v_scale_t=vsf)


def decode_matrix(codes, physical_scales):
    """Decode E2M1 nibbles times the official logical E4M3 scale, in FP32."""
    scales = logical_scales(physical_scales).float()
    require(bool(torch.isfinite(scales).all()), 'Nonfinite saved scale')
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(*codes.shape[:-1], codes.shape[-1]*2)
    return levels[nibbles.long()] * scales.repeat_interleave(16, dim=-1)


def decode_kv(packet, length):
    selected = {name: tensor[:, HEADS].contiguous()[0] for name, tensor in packet.items()}
    key_physical = decode_matrix(selected['k_fp4'], selected['k_scale'])
    inv = torch.argsort(k_permutation(length, 'cpu'))
    key = key_physical.index_select(-2, inv).contiguous()
    value = decode_matrix(selected['v_fp4_t'], selected['v_scale_t']).transpose(-2, -1).contiguous()
    return key, value


def load_packets(case):
    kv = torch.load(base.verify_file(case['kv_packets']), map_location='cpu', weights_only=True, mmap=True)
    phase0 = next(r for r in case['outputs'] if r['mode'] == 'global_mean' and r['shift'] == 0)
    q = torch.load(base.verify_file(phase0['q_packet_artifact']), map_location='cpu', weights_only=True, mmap=True)
    require(set(kv) == set(KV_KEYS) and set(q) == set(Q_KEYS), 'Saved packet fields differ')
    require(base.tree_signature(kv) == case['kv_tensor_records'], 'Saved K/V packet tensor SHA differs')
    require(base.tree_signature(q) == {name: phase0['actual_q_packets'][name] for name in Q_KEYS}, 'Saved Q packet SHA differs')
    require(q['q_fp4'].shape == (1,56,NP,64) and q['q_scale'].shape == (1,56,NP,8), 'Wrong Q packet shape')
    require(kv['k_fp4'].shape == (1,56,NP,64) and kv['k_scale'].shape == (1,56,NP,8)
            and kv['v_fp4_t'].shape == (1,56,D,NP//2) and kv['v_scale_t'].shape == (1,56,D,NP//16),
            'Wrong K/V packet shape')
    return q, kv, phase0


def verify_partition_and_decode(q, kv):
    left, right = slice_kv(kv, 0, SPLIT), slice_kv(kv, SPLIT, NP)
    for name, dim in (('k_fp4',2), ('k_scale',2), ('v_fp4_t',3)):
        require(equal_bytes(torch.cat((left[name], right[name]), dim=dim), kv[name]), 'Slice/rejoin changed packet bytes')
    tiles = []
    for piece, length in ((left, SPLIT), (right, NP-SPLIT)):
        tiles.append(piece['v_scale_t'].view(torch.uint8).reshape(1,56,D//64,length//64,256))
    joined = torch.cat(tiles, dim=3).reshape_as(kv['v_scale_t'])
    require(equal_bytes(joined, kv['v_scale_t'].view(torch.uint8)), 'V scale tile slice/rejoin changed bytes')
    key, value = decode_kv(kv, NP)
    for begin, end, piece in ((0,SPLIT,left), (SPLIT,NP,right)):
        kd, vd = decode_kv(piece, end-begin)
        require(torch.equal(kd, key[:,begin:end]) and torch.equal(vd, value[:,begin:end]),
                'CPU-decoded sliced K/V differs from full logical values')
        require(torch.equal(k_permutation(end-begin,'cpu')+begin, k_permutation(NP,'cpu')[begin:end]),
                'K physical permutation phase changed')
        del kd, vd
    qcodes = q['q_fp4'][0, HEADS].index_select(1, torch.tensor(QUERIES))
    qscale = logical_scales(q['q_scale'][0, HEADS].contiguous()).float().index_select(1, torch.tensor(QUERIES))
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    nib = torch.stack((qcodes&15, qcodes>>4), dim=-1).reshape(len(HEADS),len(QUERIES),D)
    qd = levels[nib.long()]*qscale.repeat_interleave(16, dim=-1)
    decoded = dict(q_decoded=qd, k_decoded=key[:,:N].contiguous(), v_decoded=value[:,:N].contiguous())
    require(all(bool(torch.isfinite(t).all()) for t in decoded.values()), 'Nonfinite decoded packet value')
    return decoded, dict(all_head_packet_byte_rejoin_exact=True, selected_head_logical_decode_exact=True,
                         selected_heads=list(HEADS), split=SPLIT, shard_padded_lengths=[SPLIT,NP-SPLIT],
                         shard_valid_lengths=[SPLIT,N-SPLIT], source_scale_swizzle='Official 64-row/4-column tiles')


def binding(report):
    prior_capture = complete(PRIOR/'capture_run.json')
    prior_probe = complete(PRIOR/'probe_run.json')
    complete(PRIOR/'independent_summary.json')
    frozen = complete(PRIOR/'frozen_contract.json')
    for path, sha in frozen['files'].items():
        require(base.sha256(path) == sha, f'E018 frozen source drift: {path}')
    require(prior_capture['complete_dit_calls'] == 1 and prior_probe['attention_calls'] == 27
            and prior_capture['raw_replay_exact'] and prior_capture['velocity_replay_exact'], 'Incomplete E018 execution contract')
    inherited = {**prior_capture['sources'], **prior_probe['sources']}
    base.check_sources(inherited)
    paths = set(Path(p) for p in inherited) | {Path(__file__), PLAN,
        Path(__file__).with_name('nvfp4_attention_fixed_scale.py')}
    source_records = {str(p.resolve()): base.file_record(p) for p in sorted(paths)}
    old_manifest = json.loads((ROOT/'research_state/06_experiments/E017_h3_fp4_video_manifest.json').read_text())
    require(sys.executable == old_manifest['python'], 'Unexpected Python environment')
    for name, value in old_manifest['environment'].items():
        require(os.environ.get(name) == value, f'Native environment differs: {name}')
    environment = dict(python=sys.executable, torch=torch.__version__, torch_cuda=torch.version.cuda,
        packages={name: importlib.metadata.version(name) for name in ('torch','triton','flashinfer-python')})
    require(environment['packages'] == {'torch':'2.11.0+cu128','triton':'3.6.0','flashinfer-python':'0.7.0.post1'}
            and environment['torch_cuda'] == '12.8',
            'Native package versions differ')
    require([r['block'] for r in prior_capture['cases']] == [r['block'] for r in prior_probe['cases']] == list(BLOCKS),
            'Fixed layer table changed')
    report.update(sources=source_records, environment=environment, plan=base.file_record(PLAN),
        inherited={name:base.file_record(PRIOR/name) for name in ('capture_run.json','probe_run.json','independent_summary.json','frozen_contract.json')},
        selection=dict(head_ids=list(HEADS), query_ids=list(QUERIES), frames=list(FRAMES), raster=list(RASTER),
            valid_length=N, padded_length=NP, split=SPLIT, scale=SCALE))
    return prior_capture, prior_probe


def cpu_check(probe, report):
    report['cases'] = []
    for case in probe['cases']:
        q, kv, phase0 = load_packets(case)
        decoded, proof = verify_partition_and_decode(q, kv)
        report['cases'].append(dict(block=case['block'], partition_proof=proof, decoded_tensors=base.tree_signature(decoded),
            q_packets=phase0['q_packet_artifact'], kv_packets=case['kv_packets'],
            expected_correction=phase0['actual_q_packets']['correction'], expected_bf16_output=phase0['output']['tensor']))
        del decoded, q, kv
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False)


@torch.inference_mode()
def run(args, captured, probe, report):
    checked = complete(RD/'probe_check.json')
    require(checked['cuda_initialized'] is False and checked['sources'] == report['sources']
            and checked['environment'] == report['environment'] and checked['selection'] == report['selection']
            and checked['inherited'] == report['inherited'], 'CPU/source contract changed')
    report['cpu_check'] = base.file_record(RD/'probe_check.json')
    budget(args, report)
    require(torch.cuda.get_device_capability() == (12,0), 'Expected SM120')
    torch.backends.cuda.matmul.allow_tf32 = False
    import flashinfer.nvfp4_attention_sm120 as official
    report['cases'] = []
    for capture_case, case, check in zip(captured['cases'], probe['cases'], checked['cases'], strict=True):
        budget(args, report)
        block = case['block']
        outdir = DATA/f'block{block}'
        outdir.mkdir(parents=True, exist_ok=False)
        q, kv, phase0 = load_packets(case)
        decoded, proof = verify_partition_and_decode(q,kv)
        require(base.tree_signature(decoded) == check['decoded_tensors'], 'CPU decoder drift from precheck')
        raw = torch.load(base.verify_file(capture_case['artifact']), map_location='cpu', weights_only=True, mmap=True)
        require(raw['valid_length'] == N and raw['scale'] == SCALE and raw['block'] == block, 'Capture geometry changed')
        seg = tuple(raw[key].to('cuda').transpose(0,1).unsqueeze(0).contiguous() for key in ('q','k','v'))
        processed = official._preprocess_qkv(*seg, per_block_mean=False)
        correction = processed[-1]
        correction_cpu = correction.cpu()
        require(base.tensor_record(correction_cpu) == phase0['actual_q_packets']['correction'],
                'Regenerated full GPU correction differs from E018 tensor SHA')
        correction_path = outdir/'correction.pt'
        torch.save(correction_cpu, correction_path)
        source_refs = dict(capture=capture_case['artifact'], q_packets=phase0['q_packet_artifact'], kv_packets=case['kv_packets'],
                           e018_probe=report['inherited']['probe_run.json'], correction=base.file_record(correction_path))
        score_state = dict(**decoded, correction_selected=correction_cpu[0,HEADS,:,:N].contiguous(),
            head_ids=torch.tensor(HEADS), query_ids=torch.tensor(QUERIES), scale=SCALE, valid_length=N, split=SPLIT,
            source_refs=source_refs, semantics='FP32 packet values and actual GPU FP32 correction; FP64 mathematical score material, not native accumulators')
        score_path = outdir/'score_state.pt'
        torch.save(score_state, score_path)
        row = dict(block=block, score_state=base.file_record(score_path), score_state_tensors=base.tree_signature(score_state),
            correction=source_refs['correction'], correction_tensor=base.tensor_record(correction_cpu),
            correction_e018_full_sha_exact=True, partition_proof=proof, packet_source_refs=source_refs, native={})
        report['cases'].append(row)
        del score_state, decoded, correction_cpu, processed, seg, raw
        gpu_q = {k:t.to('cuda') for k,t in q.items()}
        gpu_kv = {k:t.to('cuda') for k,t in kv.items()}
        for name in NATIVE:
            budget(args, report, before_call=True)
            begin, end, valid = (0,SPLIT,SPLIT) if name == 'fp16_left' else ((SPLIT,NP,N-SPLIT) if name == 'fp16_right' else (0,NP,N))
            actual_kv = gpu_kv if begin == 0 and end == NP else slice_kv(gpu_kv,begin,end)
            actual_correction = correction if begin == 0 and end == NP else correction[:,:,:,begin:end].contiguous()
            output_dtype = torch.bfloat16 if name == 'bf16_full' else torch.float16
            report['attention_calls'] += 1
            output, lse = official.nvfp4_attention_sm120_fwd(
                gpu_q['q_fp4'],actual_kv['k_fp4'],actual_kv['v_fp4_t'],
                gpu_q['q_scale'],actual_kv['k_scale'],actual_kv['v_scale_t'],actual_correction,
                sm_scale=SCALE,causal=False,per_block_mean=False,out_dtype=output_dtype,return_lse=True,unpadded_k_len=valid)
            torch.cuda.synchronize()
            output_cpu, lse_cpu = output.cpu(), lse.cpu()
            require(bool(torch.isfinite(output_cpu).all()) and bool(torch.isfinite(lse_cpu).all()), 'Nonfinite native output/LSE')
            valid_nhd = output_cpu[0,:,:N,:].transpose(0,1).contiguous()
            valid_record = base.tensor_record(valid_nhd)
            exact = valid_record == phase0['output']['tensor'] if name == 'bf16_full' else None
            if name == 'bf16_full':
                require(exact, 'Full BF16+LSE output differs from E018 full valid output SHA')
            sample = dict(output=output_cpu[0,HEADS].index_select(1,torch.tensor(QUERIES)).contiguous(),
                          lse=lse_cpu[0,HEADS].index_select(1,torch.tensor(QUERIES)).contiguous())
            artifact_path = outdir/f'native_{name}.pt'
            torch.save(sample,artifact_path)
            row['native'][name] = dict(artifact=base.file_record(artifact_path), sample_tensors=base.tree_signature(sample),
                full_output=base.tensor_record(output_cpu), full_lse=base.tensor_record(lse_cpu),
                valid_output_NHD=valid_record, e018_bf16_full_exact=exact, finite=True,
                kv_range=[begin,end], unpadded_k_len=valid, causal=False, per_block_mean=False,
                sm_scale=SCALE, out_dtype=str(output_dtype), return_lse=True,
                quantization='Original global Q packet and fixed globally centered K/V packet; physical slicing only')
            base.save(report,args.output)
            print(f'E019 block{block} {name} complete',flush=True)
            del output,lse,output_cpu,lse_cpu,valid_nhd,sample,actual_kv,actual_correction
        del q,kv,gpu_q,gpu_kv,correction
        gc.collect()
        torch.cuda.empty_cache()
    require(report['attention_calls'] == 12, 'Incomplete native attention allocation')
    budget(args,report)
    report.update(status='complete',cuda_initialized=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('check','run'),required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    args.output=args.output or RD/f'probe_{args.phase}.json'
    require(not args.output.exists(),'Refusing to overwrite prior report')
    torch.set_num_threads(6)
    report=dict(experiment='E019',phase=args.phase,status='running',attention_calls=0,complete_dit_calls=0,
        deadline_unix=args.deadline_unix,scope='Fixed-packet partition contract; no model inference or quality claim')
    started=time.monotonic()
    try:
        captured,probe=binding(report)
        if args.phase=='check': cpu_check(probe,report)
        else: run(args,captured,probe,report)
    except Exception:
        report.update(status='failed_stop',error=traceback.format_exc())
        raise
    finally:
        report['seconds_total']=time.monotonic()-started
        if torch.cuda.is_initialized():
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        base.save(report,args.output)
        print(json.dumps(dict(status=report['status'],report=str(args.output))),flush=True)


if __name__=='__main__':
    main()
