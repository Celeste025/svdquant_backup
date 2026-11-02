#!/usr/bin/env python3
"""Export the retained E098 NCU reports into a searchable, checked JSON bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[2]
RAW = Path('/data1/models/svdquant-wjq/research/20261014/E098')
REPORTS = {
    'sage2pp_core': RAW / 'h3_sage2pp_core_breakdown.ncu-rep',
    'sage3_core': RAW / 'h3_sage3_sm110_core_breakdown.ncu-rep',
    'sage3_source': RAW / 'h3_sage3_sm110_core_source.ncu-rep',
    'sage2pp_source': RAW / 'h3_sage2pp_core_source.ncu-rep',
}
TIMING = REPO / 'results/research/E098/sage_h3_sm110_two_scope_gpu0.json'
CAPTURE = RAW / 'h3_p1_s0_b25_480x832_73f_qkv.pt'
DEFAULT_OUTPUT = REPO / 'results/research/E098/sage_ncu_metric_bundle.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_ncu(path: Path) -> dict:
    raw = subprocess.check_output(
        ['ncu', '-i', str(path), '--page', 'raw', '--csv'], text=True
    )
    rows = list(csv.reader(io.StringIO(raw)))
    if len(rows) < 3 or len(rows[0]) != len(rows[1]) or len(rows[0]) != len(rows[2]):
        raise ValueError(f'Unexpected NCU CSV schema: {path}')
    names, units, values = rows[:3]
    fields = {}
    for name, unit, value in zip(names, units, values):
        if name in fields:
            raise ValueError(f'Duplicate NCU metric {name}: {path}')
        fields[name] = {'value': value, 'unit': unit}
    return {
        'path': str(path),
        'sha256': digest(path),
        'bytes': path.stat().st_size,
        'metric_count': len(fields),
        'fields': fields,
    }


def value(report: dict, key: str) -> float:
    return float(report['fields'][key]['value'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    timing = json.loads(TIMING.read_text())
    if timing['shape_hnd'] != [1, 56, 9482, 128] or timing['gpu'] != 'NVIDIA RTX PRO 5000 72GB Blackwell':
        raise ValueError('Wrong timing capture or GPU')
    reports = {name: read_ncu(path) for name, path in REPORTS.items() if path.exists()}
    for name in ('sage2pp_core', 'sage3_core'):
        if name not in reports:
            raise FileNotFoundError(REPORTS[name])
        arm = name.split('_')[0]
        if timing['results'][arm]['core_only']['nmse_vs_online'] != 0.0:
            raise ValueError(f'{arm} core/API agreement failed')
    expected = {'sage2pp_core': 4200, 'sage3_core': 110}
    for name, grid in expected.items():
        report = reports[name]
        if value(report, 'launch__grid_size') != grid or value(report, 'launch__sm_count') != 110:
            raise ValueError(f'Wrong native kernel launch in {name}')
        if value(report, 'sass__inst_executed_register_spilling') != 0:
            raise ValueError(f'Register spilling in {name}')
    samples = timing['results']
    summary = {}
    for arm in ('sage2pp', 'sage3'):
        source = reports[f'{arm}_core']
        summary[arm] = {
            'core_event_mean_ms': samples[arm]['core_only']['timing']['mean_ms'],
            'online_event_mean_ms': samples[arm]['online']['timing']['mean_ms'],
            'core_useful_tops': samples[arm]['core_only']['tops'],
            'online_useful_tops': samples[arm]['online']['tops'],
            'nmse_vs_bf16': samples[arm]['core_only']['nmse_vs_bf16'],
            'grid_ctas': value(source, 'launch__grid_size'),
            'threads_per_cta': value(source, 'launch__block_size'),
            'waves_per_sm': value(source, 'launch__waves_per_multiprocessor'),
            'registers_per_thread': value(source, 'launch__registers_per_thread'),
            'dynamic_shared_kbyte_per_cta': value(source, 'launch__shared_mem_per_block_dynamic'),
            'tensor_pipe_active_pct_elapsed': value(source, 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
            'sm_compute_throughput_pct_peak': value(source, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'),
            'dram_throughput_pct_peak': value(source, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'),
            'l2_hit_pct': value(source, 'lts__t_sector_hit_rate.pct'),
            'achieved_occupancy_pct_theoretical_peak': value(source, 'sm__warps_active.avg.pct_of_peak_sustained_active'),
            'eligible_warps_per_scheduler_cycle': value(source, 'smsp__warps_eligible.avg.per_cycle_active'),
            'warp_stall_wait_per_issue_ratio': value(source, 'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio'),
        }
    output = {
        'format': 'e098-sage-native-ncu-metrics-v1',
        'scope': 'real H3 prompt 1, step 0, block 25, non-causal, main 9482-token segment',
        'metric_caveat': 'NCU replay changes latency/clocks: use CUDA-event timing; pipe-active percentages are not additive stage times.',
        'timing_path': str(TIMING),
        'timing_sha256': digest(TIMING),
        'capture_path': str(CAPTURE),
        'capture_sha256': digest(CAPTURE),
        'summary': summary,
        'reports': reports,
        'missing_optional_reports': [name for name, path in REPORTS.items() if not path.exists()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'Wrote {args.output}: {len(reports)} NCU reports, '
          f'{sum(row["metric_count"] for row in reports.values())} raw metrics')


if __name__ == '__main__':
    main()
