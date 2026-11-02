#!/usr/bin/env python3
"""E141 metrics and BF16/all-sparse/first2-dense comparisons."""
from pathlib import Path
import finish_e140_draft_no_dense as finish

if __name__=='__main__':
    finish.EXPERIMENT='E141'
    finish.DATA=Path('/data1/models/svdquant-wjq/research/20261031/E141_draft_first2_dense')
    finish.DENSE_STEPS=2
    finish.COMPARISON_EXPERIMENT='E140'
    finish.WORKER_NAME='run_e141_draft_first2_dense.py'
    finish.COMPARISON_LABEL='Mixed: all steps sparse'
    finish.NEW_LABELS=['First2 dense: 70% skip /30% INT8',
                       'First2 dense: 70% skip /20% FP4 /10% INT8']
    finish.main()
