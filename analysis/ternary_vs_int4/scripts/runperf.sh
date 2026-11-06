#!/bin/bash
# unattended: select -> thread sweep -> choose threads -> main grid. Resumable (per-invocation result files).
cd /mnt/c/Users/liban/Documents/flextern
source ~/ftvenv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
L=~/exp0011/perf_run.log
python -m analysis.ternary_vs_int4.summarize select >> $L 2>&1
python -m analysis.ternary_vs_int4.perf sweep --budget-s 100000 >> $L 2>&1 || { echo PERF_FAILED_SWEEP >> $L; exit 1; }
python -m analysis.ternary_vs_int4.summarize threads >> $L 2>&1
python -m analysis.ternary_vs_int4.perf main --budget-s 100000 >> $L 2>&1 || { echo PERF_FAILED_MAIN >> $L; exit 1; }
echo PERF_ALL_DONE >> $L
