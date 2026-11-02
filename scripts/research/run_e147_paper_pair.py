#!/usr/bin/env python3
"""Generate E147 p025 with paper's 74.3/19.3/6.4 route."""
from pathlib import Path
import run_e146_text_detail_retained20 as prior
from anemoi import SparseConfig, QuantConfig

ROOT = Path(__file__).resolve().parents[2]
prior.REPORT = ROOT / 'results/research/E147'
prior.DATA = Path('/data1/models/svdquant-wjq/research/20261101/E147_paper_pair')
prior.PLAN = prior.REPORT / 'experiment_plan.md'
prior.ARMS = ('s743_fp4193_int864',)

class Router(prior.Router):
    def __init__(self, dit, arm):
        super().__init__(dit, 's80_nvfp4_10_int8_10')
        self.arm = arm
        self.sparse = SparseConfig(query_block_size=64, sparsity_ratio=.743, maxpool_weight=.2)
        self.quant = QuantConfig(nvfp4_ratio=19.3/25.7, int8_ratio=6.4/25.7,
                                 prefix_kv_precision='int8', prefix_query_precision='int8')
        self.manifest.update(arm=arm, sparse_config=vars(self.sparse), quant_config=vars(self.quant))

prior.Router = Router
if __name__ == '__main__':
    prior.main()
