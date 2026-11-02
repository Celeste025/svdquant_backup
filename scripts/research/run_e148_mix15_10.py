#!/usr/bin/env python3
"""p025: 75% skip, 15% NVFP4 and 10% INT8."""
from pathlib import Path
import run_e146_text_detail_retained20 as prior
from anemoi import SparseConfig, QuantConfig

ROOT = Path(__file__).resolve().parents[2]
prior.REPORT = ROOT / 'results/research/E148'
prior.DATA = Path('/data1/models/svdquant-wjq/research/20261101/E148_mix15_10')
prior.PLAN = prior.REPORT / 'experiment_plan.md'
prior.ARMS = ('s75_fp415_int810',)

class Router(prior.Router):
    def __init__(self, dit, arm):
        super().__init__(dit, 's80_nvfp4_10_int8_10')
        self.arm = arm
        self.sparse = SparseConfig(query_block_size=64, sparsity_ratio=.75, maxpool_weight=.2)
        self.quant = QuantConfig(nvfp4_ratio=.6, int8_ratio=.4, prefix_kv_precision='int8', prefix_query_precision='int8')
        self.manifest.update(arm=arm, sparse_config=vars(self.sparse), quant_config=vars(self.quant))

prior.Router = Router
if __name__ == '__main__':
    prior.main()
