#!/usr/bin/env python3
"""Real step0 cache -> step10 QKV replay for the same three H3 layers."""
import argparse, hashlib, json, statistics, sys, sysconfig, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
ANEMOI=ROOT/'third_party/anemoi-code';sys.path.insert(0,str(ANEMOI))
from anemoi import AttentionRouteCache,NVFP4Calibration,QuantConfig,SparseConfig,VisualLayout,anemoi_attention
def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def measure(fn,warmup,repeats):
    for _ in range(warmup):out=fn();del out
    torch.cuda.synchronize();values=[]
    for _ in range(repeats):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record();out=fn();end.record();end.synchronize();values.append(begin.elapsed_time(end));del out
    return values
@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--warmup',type=int,default=10);ap.add_argument('--repeats',type=int,default=30);a=ap.parse_args()
    assert not a.output.exists();torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest_path=ROOT/'results/research/E187/manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='complete'
    scales=json.loads((ANEMOI/'anemoi/models/minimax_h3/configs/nvfp4_tensor_scales_sm120.json').read_text())['scales']
    sparse=SparseConfig(query_block_size=128,sparsity_ratio=0.,maxpool_weight=.2)
    quant=QuantConfig(nvfp4_ratio=.9,int8_ratio=.1,prefix_kv_precision='int8',prefix_query_precision='int8')
    binary=ANEMOI/'anemoi/layers/attention/mpa'/('_cuda_sm120_q64'+sysconfig.get_config_var('EXT_SUFFIX'))
    report={'status':'running','binary_sha256':sha_file(binary),'gpu':torch.cuda.get_device_name(),
      'torch':torch.__version__,'cuda':torch.version.cuda,'manifest_sha256':sha_file(manifest_path),
      'scope':'same-layer step0 capture, step10 reuse/proxy; full API CUDA events, two reversed path orders',
      'warmup_per_round':a.warmup,'repeats_per_round':a.repeats,'layers':[]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for layer in (0,25,49):
        operands={};metadata={}
        for step in (0,10):
            entry=next(x for x in manifest['captures'] if x['step']==step and x['layer']==layer)
            path=Path(entry['path']);assert sha_file(path)==entry['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            operands[step]=[payload[name].cuda().transpose(1,2).contiguous() for name in ('q','k','v')]
            metadata[step]=(tuple(payload['video_grid']),int(payload['prefix_tokens']))
        assert metadata[0]==metadata[10]
        layout=VisualLayout(*metadata[0]);cache=AttentionRouteCache()
        cal=NVFP4Calibration(*(float(scales[k][layer]) for k in ('q_tensor_scale','k_tensor_scale','v_tensor_scale')))
        common=dict(layout=layout,layer=layer,sparse_config=sparse,quant_config=quant,calibration=cal)
        def capture():
            cache.clear();return anemoi_attention(*operands[0],**common,first_step=True,route_cache=cache)
        def reuse():return anemoi_attention(*operands[10],**common,first_step=False,route_cache=cache)
        def proxy():return anemoi_attention(*operands[10],**common)
        fingerprints={}
        for name,fn in (('capture_step0',capture),('reuse_step10',reuse),('proxy_step10',proxy)):
            out=fn();assert torch.isfinite(out).all();fingerprints[name]=tensor_sha(out);del out
        plan=cache._routes[layer].plan
        route_hashes={key:tensor_sha(getattr(plan,key)) for key in ('block_ids','packed_offsets','nvfp4_counts','fp8_counts','fp16_counts')}
        samples={name:[] for name in ('capture_step0','reuse_step10','proxy_step10')}
        functions={'capture_step0':capture,'reuse_step10':reuse,'proxy_step10':proxy}
        for order in (tuple(functions),tuple(reversed(functions))):
            for name in order:samples[name].extend(measure(functions[name],a.warmup,a.repeats))
        row={'layer':layer,'tokens':operands[0][0].size(1),'fingerprints':fingerprints,'route_hashes':route_hashes,
             'cache_bytes':cache.allocated_bytes,'timings':{name:{'mean_ms':statistics.mean(v),'median_ms':statistics.median(v),
             'std_ms':statistics.pstdev(v),'samples_ms':v} for name,v in samples.items()}}
        report['layers'].append(row);a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'layer':layer,'timings':{k:v['mean_ms'] for k,v in row['timings'].items()}},indent=2),flush=True)
        del operands,cache,plan,functions,capture,reuse,proxy;torch.cuda.empty_cache()
    report['status']='complete';a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('E188_TRAJECTORY_REPLAY_COMPLETE',flush=True)
if __name__=='__main__':main()
