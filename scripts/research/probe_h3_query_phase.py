#!/usr/bin/env python3
"""E018: query-only permutations on frozen real post-RoPE QKV.

No model forward here. K/V packets and traversal are held fixed. Global-mean
controls freeze the original centered Q and correction, not a new reduction.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import probe_h3_plain_baseline as base
import torch

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT/'results/research/E018'
DATA = Path('/data1/models/svdquant-wjq/research/20261002/E018')
PLAN = ROOT/'research_state/06_experiments/E018_h3_query_phase_plan.md'
BLOCKS, SHIFTS, MODES = (0, 24, 48), (0, 64, 128), ('bf16', 'block_mean', 'global_mean')
N, START, TOKENS, N_PAD = 22539, 1227, 21312, 22656
SCALE = 128 ** -.5
PACK_NAMES = ('q_fp4', 'k_fp4', 'v_fp4_t', 'q_scale', 'k_scale', 'v_scale_t', 'correction')
KV_IDS = (1, 2, 4, 5)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def permutation(shift, *, padded=False, device='cpu'):
    p = torch.arange(N_PAD if padded else N, device=device)
    p[START:START+TOKENS] = torch.roll(p[START:START+TOKENS], shift)
    return p

def equal_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

def budget(args, report):
    require(args.deadline_unix is not None and time.time() < args.deadline_unix,
            'Missing/expired original shared deadline')
    require(report['attention_calls'] < 27, '27-call attention budget exhausted')
    require(torch.cuda.max_memory_allocated() < 60*1024**3, 'Memory budget exceeded')

def sources():
    old = json.loads((ROOT/'results/research/E017/E017_check_block_mean.json').read_text())
    base.check_sources(old['sources'])
    paths = [Path(__file__), PLAN]
    return old['sources'] | {str(p.resolve()): base.file_record(p) for p in paths}

def cpu_check(report):
    report['geometry'] = dict(valid_length=N, video_start=START, video_tokens=TOKENS,
        padded_length=N_PAD, latent_frames=37, grid=[18,32], main_frames=[2,34])
    report['permutations'] = []
    for shift in SHIFTS:
        p = permutation(shift)
        inv = torch.argsort(p)
        require(torch.equal(p[inv], torch.arange(N)), 'Inverse query permutation failed')
        require(torch.equal(p[:START], torch.arange(START)), 'Non-video rows permuted')
        selected = torch.arange(START+2*576, START+35*576)
        moved_positions = inv[selected]
        # A 128 shift preserves every interior group's members and row order.
        if shift == 128:
            origins = p[(moved_positions//128)*128]
            require(torch.equal(origins, (selected//128)*128), '128 control changes interior groups')
        report['permutations'].append(dict(shift=shift, permutation=base.tensor_record(p),
            inverse=base.tensor_record(inv), selected_rows=int(selected.numel())))
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    report.update(status='complete', cuda_initialized=False)

def write_tensor(path, value):
    torch.save(value, path)
    return dict(artifact=base.file_record(path), tensor=base.tensor_record(value))

@torch.inference_mode()
def run(args, report):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '5', 'Only physical GPU5 is registered')
    require(torch.cuda.get_device_capability() == (12, 0), 'Expected SM120')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    checked = json.loads((RD/'probe_check.json').read_text())
    require(checked['status'] == 'complete' and checked['cuda_initialized'] is False
            and checked['sources'] == report['sources'], 'CPU/source precheck changed')
    capture_path = RD/'capture_run.json'
    captured = json.loads(capture_path.read_text())
    require(captured['status'] == 'complete', 'Capture must complete before probing')
    report['capture_reference'] = base.file_record(capture_path)
    import flashinfer.nvfp4_attention_sm120 as official
    from diffsynth.models.minimax_h3_dit import _sdpa_varlen_attention
    module = official.get_nvfp4_attention_sm120_module()
    outdir = DATA/'probe'
    outdir.mkdir(parents=True, exist_ok=False)
    report['cases'] = []
    require([c['block'] for c in captured['cases']] == list(BLOCKS), 'Capture layer selection changed')
    for case in captured['cases']:
        budget(args, report)
        path = base.verify_file(case['artifact'])
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        block = case['block']
        require(payload['block'] == block and payload['valid_length'] == N
                and payload['video_start'] == START and payload['video_tokens'] == TOKENS
                and payload['scale'] == SCALE, 'Actual captured geometry differs')
        for name in ('q','k','v','router_output'):
            t = payload[name]
            require(tuple(t.shape) == (N,56,128) and t.dtype == torch.bfloat16
                    and bool(torch.isfinite(t).all()), f'Invalid capture {name}')
        q, k, v = [payload[name].to('cuda') for name in ('q','k','v')]
        seg = tuple(t.transpose(0,1).unsqueeze(0).contiguous() for t in (q,k,v))
        packed0 = official.nvfp4_attention_sm120_quantize_qkv(*seg, per_block_mean=True)
        packet = {PACK_NAMES[i]: packed0[i].cpu() for i in KV_IDS}
        packet_path = outdir/f'block{block}_kv_packets.pt'
        torch.save(packet, packet_path)
        row = dict(block=block, capture=case['artifact'], kv_packets=base.file_record(packet_path),
            kv_tensor_records=base.tree_signature(packet), outputs=[])
        report['cases'].append(row)
        del packet
        global_pack = official.nvfp4_attention_sm120_quantize_qkv(*seg, per_block_mean=False)
        require(all(equal_bytes(global_pack[i],packed0[i]) for i in KV_IDS), 'Global K/V packets changed')
        q_global, _, _, correction = official._preprocess_qkv(*seg, per_block_mean=False)
        require(equal_bytes(correction, global_pack[-1]), 'Frozen global correction differs from official')
        q_check, s_check = torch.empty_like(global_pack[0]), torch.empty_like(global_pack[3])
        module.scaled_fp4_quant(q_global, q_check, s_check, 1)
        require(equal_bytes(q_check,global_pack[0]) and equal_bytes(s_check,global_pack[3]),
                'Direct global Q packing differs from official API')
        row['official_global_packet_parity'] = True
        del q_check, s_check, correction
        for mode in MODES:
            for shift in SHIFTS:
                budget(args, report)
                p = permutation(shift, device='cuda')
                inv = torch.argsort(p)
                before = time.monotonic()
                qp = q.index_select(0,p)
                pack_records = None
                q_packet_artifact = None
                if mode == 'bf16':
                    report['attention_calls'] += 1
                    out = _sdpa_varlen_attention(qp,k,v,torch.tensor([0,N],dtype=torch.int32),SCALE)
                else:
                    if mode == 'block_mean':
                        candidate = packed0 if shift == 0 else official.nvfp4_attention_sm120_quantize_qkv(
                            qp.transpose(0,1).unsqueeze(0).contiguous(),seg[1],seg[2],per_block_mean=True)
                        require(all(equal_bytes(candidate[i],packed0[i]) for i in KV_IDS),
                                'Query-only permutation changed official K/V packets')
                        packed = tuple(packed0[i] if i in KV_IDS else candidate[i] for i in range(7))
                        del candidate
                    else:
                        pp = permutation(shift, padded=True, device='cuda')
                        q_centered = q_global.index_select(2,pp).contiguous()
                        q_fp4, q_sf = torch.empty_like(global_pack[0]), torch.empty_like(global_pack[3])
                        module.scaled_fp4_quant(q_centered,q_fp4,q_sf,1)
                        if shift == 0:
                            require(equal_bytes(q_fp4,global_pack[0]) and equal_bytes(q_sf,global_pack[3]),
                                    'Phase-zero global pack differs from official')
                            q_fp4, q_sf = global_pack[0], global_pack[3]
                        packed = (q_fp4,packed0[1],packed0[2],q_sf,packed0[4],packed0[5],global_pack[6])
                        del q_centered, q_fp4, q_sf, pp
                    require(all(packed[i].data_ptr() == packed0[i].data_ptr() for i in KV_IDS),
                            'Consumer did not reuse original K/V storage')
                    report['attention_calls'] += 1
                    out4 = official.nvfp4_attention_sm120_fwd(*packed,sm_scale=SCALE,causal=False,
                        per_block_mean=mode=='block_mean',out_dtype=torch.bfloat16,return_lse=False,unpadded_k_len=N)
                    out = out4[0,:,:N,:].transpose(0,1).contiguous()
                    # Actual used packet hashes; scale storage is swizzled and never rolled.
                    pack_records = {PACK_NAMES[i]:base.tensor_record(packed[i]) for i in (0,3,6)}
                    qp_path = outdir/f'block{block}_{mode}_shift{shift}_q_packets.pt'
                    torch.save({PACK_NAMES[i]:packed[i].cpu() for i in (0,3)},qp_path)
                    q_packet_artifact = base.file_record(qp_path)
                    del out4, packed
                torch.cuda.synchronize()
                restored = out.index_select(0,inv).cpu()
                require(bool(torch.isfinite(restored).all()), 'Nonfinite attention output')
                replay = None
                if mode == 'block_mean' and shift == 0:
                    replay = equal_bytes(restored,payload['router_output'])
                    require(replay, 'Unmodified block attention differs from original model capture')
                record = dict(mode=mode,shift=shift,permutation=base.tensor_record(p.cpu()),
                    output=write_tensor(outdir/f'block{block}_{mode}_shift{shift}.pt',restored),
                    actual_q_packets=pack_records,fixed_kv_storage=mode!='bf16',
                    q_packet_artifact=q_packet_artifact,
                    original_router_output_exact=replay,seconds_including_io=time.monotonic()-before)
                row['outputs'].append(record)
                base.save(report,args.output)
                print(f'E018 block{block} {mode} shift{shift} saved',flush=True)
                del out, restored, qp, p, inv
        del q,k,v,seg,packed0,global_pack,q_global,payload
        gc.collect()
        torch.cuda.empty_cache()
    require(report['attention_calls'] == 27, 'Incomplete attention allocation')
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    report.update(status='complete', complete_dit_calls=0, cuda_initialized=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['check','run'],required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    args.output = args.output or RD/f'probe_{args.phase}.json'
    require(not args.output.exists(), 'Refusing to overwrite prior report')
    RD.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    report = dict(experiment='E018',phase=args.phase,status='running',attention_calls=0,
        sources=sources(),deadline_unix=args.deadline_unix)
    try:
        if args.phase == 'check':
            cpu_check(report)
        else:
            run(args,report)
    except BaseException as exc:
        report.update(status='failed_stop',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['seconds_total'] = time.monotonic()-started
        base.save(report,args.output)

if __name__ == '__main__':
    main()
