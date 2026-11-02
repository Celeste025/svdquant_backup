#!/usr/bin/env python3
"""Extract comparable GPU2 Sage2++/corrected-Sage3 NCU core counters."""
from __future__ import annotations

import json
from pathlib import Path

from export_e098_sage_ncu import read_ncu

ROOT = Path(__file__).resolve().parents[2]
RAW = Path('/data1/models/svdquant-wjq/research/20261025/E130')
REPORTS = {'sage2pp': RAW / 'sage2pp_core_gpu2.ncu-rep',
           'sage3_maskfix': RAW / 'sage3_maskfix_core_gpu2.ncu-rep',
           'sage3_unfixed': RAW / 'sage3_unfixed_core_gpu2.ncu-rep',
           'sage3_maskbit': RAW / 'sage3_maskbit_core_gpu2.ncu-rep',
           'sage3_tailguard': RAW / 'sage3_tailguard_core_gpu2.ncu-rep',
           'sage3_stage2': RAW / 'sage3_stage2_core_gpu2.ncu-rep',
           'sage3_m128s1': RAW / 'sage3_m128s1_core_gpu2.ncu-rep'}
METRICS = {
    'grid_ctas': 'launch__grid_size',
    'sm_count': 'launch__sm_count',
    'waves_per_sm': 'launch__waves_per_multiprocessor',
    'threads_per_cta': 'launch__block_size',
    'registers_per_thread': 'launch__registers_per_thread',
    'dynamic_shared_kb': 'launch__shared_mem_per_block_dynamic',
    'spilling_requests': 'sass__inst_executed_register_spilling',
    'tensor_pipe_pct_elapsed': 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm_throughput_pct_elapsed': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram_throughput_pct_elapsed': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'l2_sector_hit_pct': 'lts__t_sector_hit_rate.pct',
    'achieved_occupancy_pct': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'eligible_warps_per_scheduler_cycle': 'smsp__warps_eligible.avg.per_cycle_active',
    'warp_stall_wait_per_issue': 'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',
    'warp_stall_sleeping_per_issue': 'smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio',
    'warp_stall_short_scoreboard_per_issue': 'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
    'warp_stall_math_pipe_per_issue': 'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio',
    'warp_stall_long_scoreboard_per_issue': 'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'warp_stall_branch_per_issue': 'smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio',
}


def main():
    reports = {name: read_ncu(path) for name, path in REPORTS.items()}
    summary = {}
    for name, report in reports.items():
        fields = report['fields']
        summary[name] = {alias: {'value': fields[key]['value'], 'unit': fields[key]['unit']}
                         if key in fields else None for alias, key in METRICS.items()}
        summary[name]['report_sha256'] = report['sha256']
        summary[name]['report_path'] = report['path']
        summary[name]['raw_metric_count'] = report['metric_count']
        if float(summary[name]['sm_count']['value']) != 110:
            raise RuntimeError(f'{name}: profile is not RTX PRO 5000')
    result = {'format': 'e130-sage2pp-sage3-core-ncu-v1',
              'scope': 'same real H3 [1,56,9482,128] QKV, noncausal Sage3 before/after tail-mask fix, GPU2',
              'caveat': 'NCU replay perturbs duration; compare CUDA-event time separately. Pipe percentages overlap and are not additive.',
              'summary': summary}
    path = ROOT / 'results/research/E130/ncu_core_summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
