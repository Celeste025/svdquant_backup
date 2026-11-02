#!/usr/bin/env python3
"""E162: E159 proxy with every main block/step at 90% NVFP4 + 10% INT8."""
import argparse
import json
from pathlib import Path

import torch
import run_e159_route_reuse_whitebox as prior
import run_e150_whitebox_bf16 as baseline
from anemoi import SparseConfig, QuantConfig, VisualLayout, anemoi_attention

ROOT = Path(__file__).resolve().parents[2]


class Router(prior.Router):
    dense_first_steps = 0
    dense_first_layers = 0

    def __init__(self, dit, arm):
        super().__init__(dit, arm)
        self.mode = 'proxy_new'
        self.sparse = SparseConfig(query_block_size=128, sparsity_ratio=0., maxpool_weight=.2)
        self.quant = QuantConfig(nvfp4_ratio=.9, int8_ratio=.1,
                                 prefix_kv_precision='int8', prefix_query_precision='int8')
        self.manifest.update(mode='proxy_new', sparse_config=vars(self.sparse),
                             quant_config=vars(self.quant), dense_first_steps=0,
                             dense_first_layers=0, first_step_policy='none; all 20 steps and all 50 main blocks mixed',
                             linears='BF16')

    def _dispatch(self, q, k, v, cu_seqlens, softmax_scale):
        log = self._active_log.get()
        block = self._active_main.get()
        expected = log.expected_refiner_cu if block is None else log.expected_cu
        assert softmax_scale == 128 ** -.5
        assert q.shape == k.shape == v.shape == (expected[-1], 56, 128)
        log.remember_cu(cu_seqlens, expected, 'refiner' if block is None else 'main')
        if block is None:
            log.refiner_helpers += 1
            return self.original_helper(q, k, v, cu_seqlens, softmax_scale)
        n, total = expected[1:]
        tensors = [t[:n].unsqueeze(0).contiguous() for t in (q, k, v)]
        result = anemoi_attention(*tensors, layout=VisualLayout(self.grid, self.prefix),
                                  layer=block, sparse_config=self.sparse, quant_config=self.quant,
                                  calibration=prior.prior.calibration(block))
        out = torch.empty_like(q)
        out[:n] = result[0]
        if total > n:
            out[n:] = self.original_helper(q[n:], k[n:], v[n:], log.padding_cpu_cu, softmax_scale)
        log.main_rows.append(dict(block=block, path='draft', capture=False,
                                  prefix_tokens=self.prefix, valid_length=n, padding_length=total-n))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt-id', type=int, choices=(25, 48, 44, 7), required=True)
    args = ap.parse_args()
    prior.MODE = 'proxy'
    bf = json.loads((ROOT / f'results/research/E150/p{args.prompt_id:03d}/bf16_denoise.json').read_text())
    assert bf['status'] == 'denoise_complete' and Path(bf['terminal_latents']).exists()
    baseline.REPORT = ROOT / 'results/research/E162/proxy_new'
    baseline.DATA = ROOT / 'results/local/E162/proxy_new'
    baseline.ARM = 'proxy_new'
    baseline.e121.Router = Router
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    read_prompt = baseline.read_prompt
    baseline.read_prompt = lambda pid: read_prompt(pid, Path('/home/wjq/workspace/178866172854036'))
    baseline.denoise(args.prompt_id)
    result_path = baseline.REPORT / f'p{args.prompt_id:03d}/bf16_denoise.json'
    result = json.loads(result_path.read_text())
    assert result['initial_latent_sha256'] == bf['initial_latent_sha256']
    x = torch.load(result['terminal_latents'], map_location='cpu', weights_only=True)['video_latents'].float()
    y = torch.load(bf['terminal_latents'], map_location='cpu', weights_only=True)['video_latents'].float()
    assert x.shape == y.shape
    result['comparison'] = {
        'bf16_reference': bf['terminal_latents'],
        'video_latent_nmse': float((x-y).double().square().sum() / y.double().square().sum()),
        'initial_noise_matches_bf16': True,
        'mode': 'proxy_new',
    }
    assert len(result['steps']) == 20
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"COMPLETE p{args.prompt_id:03d} NMSE={result['comparison']['video_latent_nmse']:.8f}", flush=True)


if __name__ == '__main__':
    main()
